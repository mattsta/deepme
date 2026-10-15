"""The proxy logic the balancer and the gateway must never disagree about.

Both services answer the same question — "which backend holds this job?" —
and both got it wrong in the same way once: first-non-400-wins, which served
one caller's video for another caller's id, because engines answer for ids
they did not mint and replica-minted ids are not namespaced across fleets.

The fix was applied TWICE and then DRIFTED. The balancer learned to count
unreachable backends, to treat an incomplete poll as inconclusive, and to
raise on a conflict so the caller sees 409; the gateway kept returning None
for every one of those cases, which its caller renders as 404 — telling a
client the id is unknown and therefore safe to resubmit, which is the exact
opposite of the truth and bills the generation a second time. Adding a
modality meant editing two tuples in two files, and the guard's own argument
("ask EVERY backend") held in one file and not the other.

So the rule lives here once, parameterized by the only things that actually
differ: which backends to ask, and what to do when one is found.

DEPLOYMENT: the balancer and gateway are pushed as FLAT FILES and import
their siblings by bare name (`from proxycore import ...`), so this module is
copied into BOTH remote directories by balancerctl and gatewayctl. It must
import nothing from `deepme` — on the server there is no such package.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import httpx

# Job-scoped routes: "/v1/videos/{id}", ".../content". The id is the segment
# straight after the collection name. ONE tuple: a modality added here is
# routable by both services at once, which is what stopped images and meshes
# from being half-supported in whichever file someone remembered to edit.
COLLECTIONS = ("videos", "images", "meshes")

# The balancer stamps every job-scoped response with the EFFECTIVE job id.
# It is authoritative where a body is not: a handed-over TICKET poll is
# rewritten to the real job, so its body carries the job's id and never the
# probed ticket id — matching on the body alone found zero claimants and
# 404'd a live, billable job, which the client then resubmitted.
JOB_ID_HEADER = "x-deepme-job-id"

TERMINAL_STATUSES = ("completed", "failed", "cancelled", "error")


class AmbiguousOwner(Exception):
    """Two or more backends verifiably claim one id, or may.

    The resource exists TWICE (or we could not rule that out), which is a
    CONFLICT. Reporting it as 404 tells a client the id is unknown and
    therefore safe to resubmit — the opposite of the truth, and it bills the
    work again. Callers map this to 409.
    """


def auth_headers(request: Any, api_key: str) -> dict[str, str]:
    """Forward the caller's credentials, or our own if it sent none."""
    incoming = request.headers.get("authorization")
    if incoming:
        return {"authorization": incoming}
    return {"authorization": f"Bearer {api_key}"} if api_key else {}


def job_id_of(path: str) -> str | None:
    """The job id in a job-scoped path, or None if the path is not one."""
    parts = [p for p in path.split("/") if p]
    for i, part in enumerate(parts):
        if part in COLLECTIONS and i + 1 < len(parts):
            return parts[i + 1]
    return None


def collection_of(path: str) -> str:
    """Which API family a request belongs to, for reconcile probes."""
    for part in path.split("/"):
        if part in COLLECTIONS:
            return part
    return "videos"


def status_path(path: str, job_id: str) -> str:
    """The cheap status URL for a job, whatever sub-resource was asked for.

    Ownership must never be probed with the caller's own path: for
    ".../{id}/content" that downloads the whole MP4 from every backend —
    buffered, in full — just to find out which one has it.

    Rebuilt by SEGMENTS, never substring surgery: partition on "/{id}"
    matched INSIDE "/v1" for short ids ("/v1/videos/1" -> "/v/1"), and a
    corrupted probe path 404s everywhere — the result stranded.
    """
    parts = path.split("/")
    for i, part in enumerate(parts):
        if part in COLLECTIONS and i + 1 < len(parts):
            return "/".join([*parts[: i + 1], job_id])
    return f"/v1/videos/{job_id}"


def safe_headers(headers: Any) -> dict[str, str]:
    """Response headers safe to pass through a proxy hop.

    content-length and the encoding/framing headers describe the UPSTREAM
    body; re-sending them alongside a re-framed (and, for streams, decoded)
    body makes the response undecodable downstream.
    """
    drop = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


def content_headers(request: Any) -> dict[str, str]:
    ct = request.headers.get("content-type")
    return {"content-type": ct} if ct else {}


def is_content(path: str) -> bool:
    """A request for BYTES, as opposed to a status read."""
    return path.rstrip("/").endswith("/content")


def _claims(response: httpx.Response, job_id: str) -> bool:
    """Does this response VERIFIABLY prove the backend holds `job_id`?

    A 200 alone proves nothing — that assumption is what served another
    job's bytes. The id must be NAMED, by the authoritative header or by the
    body.
    """
    if response.status_code != 200:
        return False
    if response.headers.get(JOB_ID_HEADER) == job_id:
        return True
    with contextlib.suppress(ValueError, TypeError, KeyError):
        return str(dict(response.json()).get("id") or "") == job_id
    return False


def _status_of(response: httpx.Response) -> str:
    with contextlib.suppress(ValueError, TypeError, KeyError):
        return str(dict(response.json()).get("status") or "")
    return ""


async def find_owner(
    job_id: str,
    path: str,
    headers: dict,
    *,
    candidates: list[str],
    client: httpx.AsyncClient,
    log: Callable[[str], None],
    on_ambiguous: Callable[[str, int, int], None] | None = None,
) -> tuple[str, str] | None:
    """Which backend holds this job? Returns (url, status), or None.

    Raises AmbiguousOwner when the answer is a CONFLICT rather than an
    absence. Only needed after a service lost its mapping; sequential and
    cheap, because a status GET is milliseconds and a wrong guess would
    strand — or misdeliver — the result.

    `candidates` must be EVERY attached backend, including ones currently
    marked unhealthy. Filtering them defeats the guard's own argument: when
    the TRUE owner was briefly unhealthy (a blip — revival exists because
    blips happen) and another backend held a stale colliding id, the probe
    saw exactly one claimant, concluded "unambiguous", and bound the job to
    the wrong backend with full confidence.
    """
    probe = status_path(path, job_id)
    claimants: list[tuple[str, str]] = []
    unreachable = 0
    for url in candidates:
        try:
            r = await client.get(f"{url}{probe}", headers=headers)
        except httpx.HTTPError:
            unreachable += 1
            continue
        if _claims(r, job_id):
            claimants.append((url, _status_of(r)))

    if len(claimants) > 1 or (claimants and unreachable):
        log(
            f"owner probe ambiguous for {job_id}: {len(claimants)} claimant(s), "
            f"{unreachable} backend(s) unreachable — refusing to guess"
        )
        if on_ambiguous is not None:
            on_ambiguous(job_id, len(claimants), unreachable)
        raise AmbiguousOwner(job_id)
    if len(claimants) != 1 or unreachable:
        # An INCOMPLETE poll is not an unambiguous one: a backend we could
        # not ask may be the real owner.
        if unreachable:
            log(f"owner probe inconclusive for {job_id}: {unreachable} backend(s) unreachable")
        return None
    return claimants[0]
