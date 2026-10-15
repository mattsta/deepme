"""Content reconciliation for the duplicate/misattribution incident.

Ghost generations (submission timeouts that still reached an engine) and
identity bugs left batches holding videos that do not match their prompts,
with byte-identical copies spread across items and batches. The recovery the
operator chose: hash every clip, keep only the OLDEST copy of any duplicated
content, delete the newer copies, and reset the items that lost their file
to PENDING — with job_id CLEARED, so the next run regenerates from the
prompt instead of re-adopting the same wrong server job. That reset is the
whole point: deleting the file alone would make resume re-download the very
misattribution being repaired.

Pure planning is separated from execution so a dry run IS the plan.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import batch as batch_mod
from .digests import DigestCache

MEDIA_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".wav", ".png", ".jpg")

# What an item that names no duration actually asks for (h3-video's
# [generate] target.duration_seconds). Used ONLY to normalise request
# identity — an omitted field and the default it resolves to are the same
# request, and hashing them apart is what made four copies of one clip look
# like four different requests with impossibly identical bytes.
DEFAULT_SECONDS = 5


def census_key(item: batch_mod.Item) -> str:
    """Canonical request identity: what the engine was ASKED to make.

    Everything that shapes the output goes in — prompt, seed (None
    included), steps, seconds, params. Two items sharing a key made the
    same request, whether or not either has produced bytes yet.
    """
    payload = json.dumps(
        {
            "prompt": item.prompt,
            "seed": item.seed,
            "steps": item.steps,
            # NORMALISED, because an omitted field is not a different
            # request — it is the same request relying on the default. The
            # identical prompt appears in four backlogs, three of which
            # omit `seconds` while one writes `"seconds": 5`, which IS the
            # engine's default. Hashing them apart made four copies of one
            # clip look like four different requests that had somehow
            # produced identical bytes — i.e. misattribution — so they were
            # deleted and regenerated on every pass.
            "seconds": item.seconds or DEFAULT_SECONDS,
            "params": item.params,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def item_spec_key(item: batch_mod.Item) -> str:
    """The REQUEST this item makes. Same key => identical output is expected.

    Seedless items count too, and that correction matters more than any
    other line here. The engine defaults an omitted seed deterministically,
    so two items with the same prompt produce the SAME BYTES whether or not
    a seed was written down — across batches, months apart.

    Requiring an explicit seed made this return None for every plain-prompt
    batch, so the "expected duplicate" branch could never fire: legitimate
    determinism was classified as misattribution, the file was deleted and
    the item reset, the next run regenerated the identical bytes, and the
    next dedupe deleted it again. Hours of GPU time went into that loop.
    Identity of a request is the request, not whether someone typed a seed.
    """
    return census_key(item)


@dataclass(frozen=True)
class SpecMember:
    batch: str
    index: int
    status: str


@dataclass
class SpecTwins:
    """Items that made the exact same request — expected duplicates by
    construction, whether or not their bytes exist yet."""

    key: str
    prompt: str
    seed: int | None
    members: list[SpecMember]

    @property
    def deterministic(self) -> bool:
        """Will re-running these reproduce the same bytes?

        YES either way, which is the correction: the engine defaults an
        omitted seed deterministically, so identical prompts give identical
        output whether or not a seed was written down. Treating seedless
        twins as "every run differs" is what made the platform delete
        legitimate duplicates and regenerate them forever.
        """
        return True


def existing_by_spec() -> dict[str, tuple[str, int, Path]]:
    """request key -> (batch, item index, file), for every clip on disk.

    The pre-generation duplicate guard reads this. It answers "does this
    platform already hold the output of this exact request?", which is the
    only question that can END the delete/regenerate loop: deleting an
    output never fixed a duplicate REQUEST, because the request is still in
    a prompts file and the next run remakes it.

    OLDEST WINS, matching the content dedupe's rule, so the survivor is
    stable across runs — if a later copy could displace an earlier one the
    two tools would disagree about which file is canonical and each would
    undo the other.

    EVERY batch is indexed, including the caller's own. Duplicates live
    inside a single backlog as often as across several — 64 pending items in
    one batch were twins of content that batch had already produced — so
    excluding the whole batch would have skipped exactly that case. The
    caller identifies its own item by (batch, index) instead, which is the
    only thing that is genuinely not a duplicate of itself.
    """
    found: dict[str, tuple[str, int, Path]] = {}
    for b in load_batches():
        out = Path(b.out_dir)
        for item in b.items:
            if item.status != batch_mod.DONE or not item.file:
                continue
            path = out / item.file
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue  # recorded done, file gone: not something to point at
            key = census_key(item)
            prior = found.get(key)
            if prior is not None:
                with contextlib.suppress(OSError):
                    if prior[2].stat().st_mtime <= mtime:
                        continue
            found[key] = (b.name, item.index, path)
    return found


def load_batches() -> list[batch_mod.Batch]:
    """Every READABLE batch on disk — the one load feeding every analysis.

    A truncated or schema-drifted state.json raises out of `Batch.load`;
    letting that escape made one damaged batch take down the census and
    dedupe for ALL batches, during exactly the incident these tools exist
    for. Unreadable batches are skipped and named by the caller.
    """
    out = []
    for state in sorted(batch_mod.BATCH_DIR.glob("*/state.json")):
        with contextlib.suppress(ValueError, TypeError, KeyError, OSError):
            b = batch_mod.Batch.load(state.parent.name)
            if b is not None:
                out.append(b)
    return out


@dataclass
class Suspicion:
    """One item whose delivered content may not be its own."""

    batch: str
    index: int
    kind: str
    detail: str


def audit_batch(b: batch_mod.Batch) -> list[Suspicion]:
    """Cross-examine a batch's own records for misattribution.

    Everything here is checkable from disk — no server, no hashing of the
    whole archive — and each check answers a specific way an item can end
    up holding another prompt's output:

    * `stale-adoption`  the item carries a job id but is RUNNING with no
      live runner. A resume passes that id to `adopt=`, which SKIPS
      submission entirely and downloads whatever that id now resolves to.
      If the id is old, the item receives OLD content for a NEW prompt.
    * `shared-job-id`   two items name the SAME server job, so at most one
      of them can legitimately own its bytes.
    * `prompt-drift`    the journal recorded this item under a different
      prompt than the state file now holds — i.e. the prompts FILE changed
      under a batch that matches items by INDEX, so index N no longer
      means what it meant when job N was submitted.
    * `missing-file`    marked done, file absent.
    """
    out: list[Suspicion] = []
    by_job: dict[str, list[int]] = {}
    outdir = Path(b.out_dir)
    unverifiable: list[int] = []
    for item in b.items:
        if item.job_id:
            by_job.setdefault(item.job_id, []).append(item.index)
        # A job id with no request fingerprint predates the binding and
        # cannot be proven to belong to this item. Adoption TRUSTS it —
        # disowning every one would order the whole archive regenerated —
        # so the honest thing is to say so rather than let it pass silently.
        if item.job_id and item.job_prompt is None:
            unverifiable.append(item.index)
        if item.status == batch_mod.RUNNING and item.job_id:
            out.append(
                Suspicion(
                    b.name,
                    item.index,
                    "stale-adoption",
                    f"RUNNING with job {item.job_id} — a resume will ADOPT it "
                    f"and download whatever it resolves to, without generating",
                )
            )
        if item.status == batch_mod.DONE and item.file and not (outdir / item.file).exists():
            out.append(
                Suspicion(b.name, item.index, "missing-file", f"done but {item.file} is absent")
            )
    for job_id, indexes in sorted(by_job.items()):
        if len(indexes) > 1:
            out.append(
                Suspicion(
                    b.name,
                    indexes[0],
                    "shared-job-id",
                    f"items {indexes} all name job {job_id} — at most one can own its bytes",
                )
            )
    if unverifiable:
        shown = ", ".join(f"{i:04d}" for i in unverifiable[:8])
        out.append(
            Suspicion(
                b.name,
                unverifiable[0],
                "unverifiable-binding",
                f"{len(unverifiable)} item(s) carry a job id recorded before "
                f"ids were bound to requests ({shown}"
                f"{'…' if len(unverifiable) > 8 else ''}) — adoption trusts "
                f"them; regenerate the batch if its outputs are suspect",
            )
        )
    out.extend(_prompt_drift(b))
    return out


def _prompt_drift(b: batch_mod.Batch) -> list[Suspicion]:
    """Items whose prompt changed after a job was recorded against them.

    The runner matches items to the prompts file BY INDEX. If that file is
    edited or regenerated, index N keeps the job id submitted for the OLD
    prompt — and a resume adopts it, delivering the old prompt's video as
    the new prompt's output. The journal is the only record of which
    prompt a job was actually submitted for.
    """
    path = batch_mod.BATCH_DIR / b.name / "journal.jsonl"
    if not path.exists():
        return []
    seen: dict[int, str] = {}
    with contextlib.suppress(OSError):
        for line in path.read_text().splitlines():
            with contextlib.suppress(ValueError):
                rec = json.loads(line)
                if rec.get("event") == "started" and "prompt" in rec:
                    seen[int(rec["item"])] = str(rec["prompt"])
    out = []
    for item in b.items:
        was = seen.get(item.index)
        if was is not None and was != item.prompt:
            out.append(
                Suspicion(
                    b.name,
                    item.index,
                    "prompt-drift",
                    f"journal recorded {was[:40]!r}, state now holds {item.prompt[:40]!r}",
                )
            )
    return out


def unreadable_batches() -> list[str]:
    """Names of batches whose state cannot be parsed — a fact worth showing,
    since `load_batches` deliberately skips them."""
    bad = []
    for state in sorted(batch_mod.BATCH_DIR.glob("*/state.json")):
        try:
            batch_mod.Batch.load(state.parent.name)
        except Exception:
            bad.append(state.parent.name)
    return bad


@dataclass
class CensusSummary:
    """The GPU verdict, as three disjoint answers to "should this run?".

    already_generated -> pending seeded twins whose exact bytes EXIST in a
                         twin's slot: running them burns a GPU-hour to
                         recreate a file we refuse to keep twice. Never run.
    redundant         -> seeded twin sets with NO bytes yet: one member of
                         each set is real work, the rest recreate it.
                         Run one; never run the others.
    seedless          -> pending duplicate REQUESTS without seeds: each run
                         answers differently, so running them is an intent
                         decision, not waste detection.

    Pure report — nothing here mutates state or files.
    """

    already_generated: list[SpecMember] = field(default_factory=list)
    redundant: list[SpecMember] = field(default_factory=list)
    seedless: list[SpecMember] = field(default_factory=list)
    # (batch, index) -> the twin that makes it skippable, human-readable.
    # The reason travels INTO the state file on fixup, so a skipped item
    # explains itself forever.
    sources: dict[tuple[str, int], str] = field(default_factory=dict)

    def by_batch(self, members: list[SpecMember]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for m in members:
            out.setdefault(m.batch, []).append(m.index)
        return {k: sorted(v) for k, v in sorted(out.items())}


def summarize_census(census: list[SpecTwins]) -> CensusSummary:
    """Reduce the census to the run/never-run verdict. Pure."""
    summary = CensusSummary()
    for group in census:
        pending = [m for m in group.members if m.status in ("pending", "failed")]
        if not pending:
            continue
        if not group.deterministic:
            summary.seedless.extend(pending)
            for m in pending:
                others = [x for x in group.members if x is not m]
                summary.sources[(m.batch, m.index)] = (
                    f"seedless duplicate request of {_who(others[0])}"
                )
        elif (done := next((m for m in group.members if m.status == "done"), None)) is not None:
            summary.already_generated.extend(pending)
            for m in pending:
                summary.sources[(m.batch, m.index)] = f"twin of generated {_who(done)}"
        else:
            keeper = pending[0]
            summary.redundant.extend(pending[1:])
            for m in pending[1:]:
                summary.sources[(m.batch, m.index)] = f"twin of pending {_who(keeper)}"
    return summary


def _who(m: SpecMember) -> str:
    return f"{m.batch}[{m.index:04d}]"


def apply_fixup(summary: CensusSummary, *, skip_seedless: bool = False) -> dict[str, int]:
    """Mark every duplicate generation SKIPPED in its own state file.

    A skipped item is invisible to `batch run` (todo() never selects it),
    carries its reason in `error`, and is journaled — the backlog that
    remains after this IS the unique work, visible in `batch list`.
    Returns {batch name: items skipped}.
    """
    targets = list(summary.already_generated) + list(summary.redundant)
    if skip_seedless:
        targets += list(summary.seedless)
    by_name: dict[str, list[SpecMember]] = {}
    for m in targets:
        by_name.setdefault(m.batch, []).append(m)
    out: dict[str, int] = {}
    for name, members in sorted(by_name.items()):
        if batch_mod.Batch.load(name) is None:
            continue
        # Under the batch's own claim, and re-read INSIDE it: a live runner
        # holds its state in memory and saves after every item, so both an
        # unclaimed edit and an edit built on a pre-lock snapshot are
        # silently reverted by the runner's next save.
        with batch_mod.claim(batch_mod.Batch(name=name, profile="", items=[], out_dir="")):
            b = batch_mod.Batch.load(name)
            if b is None:
                continue
            items = {i.index: i for i in b.items}
            for m in members:
                item = items.get(m.index)
                if item is None or item.status not in (batch_mod.PENDING, batch_mod.FAILED):
                    continue
                item.status = batch_mod.SKIPPED
                item.job_id = None
                item.error = summary.sources.get((m.batch, m.index), "duplicate request")
                batch_mod.journal(b, "skipped", item=m.index, reason=item.error)
                out[name] = out.get(name, 0) + 1
            b.save()
    return out


def spec_census(batches: list[batch_mod.Batch]) -> list[SpecTwins]:
    """Request-identity groups with 2+ members, across all batches.

    Pure state analysis — no files touched — so it also names the
    duplicates that WILL exist once pending items generate.
    """
    groups: dict[str, SpecTwins] = {}
    for b in batches:
        for item in b.items:
            key = census_key(item)
            group = groups.setdefault(
                key, SpecTwins(key=key, prompt=item.prompt, seed=item.seed, members=[])
            )
            group.members.append(SpecMember(b.name, item.index, item.status))
    twins = [g for g in groups.values() if len(g.members) >= 2]
    return sorted(twins, key=lambda g: (-len(g.members), g.prompt))


@dataclass
class DupGroup:
    """One set of byte-identical clips. The oldest survives; the rest go.

    MD5 IS THE GROUND TRUTH. Identical bytes are a duplicate file, and a
    duplicate file is deleted — there is no case where keeping a second copy
    of the same bytes is right. What differs between copies is only what
    should happen to the ITEM that lost its file, which is `skip` vs `reset`.
    """

    digest: str
    keep: Path
    drop: list[Path]


@dataclass
class Plan:
    """Everything the apply step would do, computable without doing it."""

    scanned: int = 0
    groups: list[DupGroup] = field(default_factory=list)
    # (batch, index, path deleted) -> PENDING. The file was NOT this item's
    # honest output: its request differs from the keeper's, and a
    # deterministic engine cannot make one video from two requests. It must
    # regenerate to produce the correct (different) bytes.
    resets: list[tuple[str, int, str]] = field(default_factory=list)
    # (batch, index, path deleted, the surviving file) -> SKIPPED. Same
    # request as the keeper, so the keeper's file IS this item's output.
    # Regenerating would recreate bytes we already hold — that is the loop.
    skips: list[tuple[str, int, str, str]] = field(default_factory=list)

    @property
    def drops(self) -> list[Path]:
        return [p for g in self.groups for p in g.drop]


def scan_dirs(dirs: list[Path], cache: DigestCache | None = None) -> dict[str, list[Path]]:
    """digest -> files carrying that content, across every directory.

    Only flat media files: state.json, frames/ and other bookkeeping are
    not clips and must never be deleted by content policy.

    Digests come through the shared cache, so a 20th run of the repair
    re-hashes nothing that has not changed — and batch downloads primed
    the cache already.
    """
    cache = cache or DigestCache.load()
    by_digest: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            rp = p.resolve()
            if rp in seen or not p.is_file() or p.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            seen.add(rp)
            by_digest.setdefault(cache.digest(p), []).append(p)
    cache.save()
    return by_digest


def build_plan(
    by_digest: dict[str, list[Path]],
    items_by_path: dict[Path, tuple[str, int, str | None]],
) -> Plan:
    """Byte-identical clips: keep the oldest, delete the rest.

    MD5 DECIDES DELETION. Identical bytes are a duplicate file and one copy
    is enough, always. An earlier version SPARED copies whose request matched
    the keeper's, because deleting a file then reset its item to PENDING and
    the next run regenerated it — a delete/regenerate loop. Sparing was a
    workaround for that loop, not a judgement about the file, and it meant
    `dedupe --apply` reported duplicates and then deleted nothing.

    The loop is now closed where it belongs (batch.skip_existing refuses to
    generate a request whose output exists), so deletion is safe again and
    the spec is used for the only question left: what should happen to the
    ITEM that just lost its file?

      same request as the keeper -> SKIPPED. The keeper's file IS this
          item's output; regenerating would recreate bytes we hold.
      different request          -> PENDING. A deterministic engine cannot
          make one video from two requests, so this file was never that
          item's honest output. It must regenerate to produce the correct,
          different bytes.
      no owning item             -> the file is just deleted; nothing to say.

    `items_by_path` maps a clip path to (batch, item index, spec key).
    """
    plan = Plan(scanned=sum(len(v) for v in by_digest.values()))
    for digest, paths in sorted(by_digest.items()):
        # A file can vanish between the scan and here (a runner replacing an
        # item's clip, another sweep). Drop it from the group rather than
        # letting one FileNotFoundError abort the whole plan.
        dated = []
        for p in paths:
            with contextlib.suppress(OSError):
                dated.append((p.stat().st_mtime, str(p), p))
        if len(dated) < 2:
            continue
        # Oldest mtime survives; path as the deterministic tiebreak.
        ordered = [p for _m, _s, p in sorted(dated, key=lambda t: (t[0], t[1]))]
        keep, rest = ordered[0], ordered[1:]

        # EVERY newer copy is deleted. No exceptions, no sparing: identical
        # bytes are a duplicate file.
        plan.groups.append(DupGroup(digest=digest, keep=keep, drop=list(rest)))
        keeper_spec = (items_by_path.get(keep.resolve()) or (None, None, None))[2]
        for p in rest:
            owner = items_by_path.get(p.resolve())
            if not owner:
                continue  # unowned file: delete it, no item to update
            batch_name, index, spec = owner
            if spec is not None and spec == keeper_spec:
                # Same request: the keeper's file IS this item's output.
                plan.skips.append((batch_name, index, str(p), str(keep)))
            else:
                # Different (or unknowable) request: this file was never
                # this item's honest output. Regenerate the correct bytes.
                plan.resets.append((batch_name, index, str(p)))
    return plan


def batch_clip_index(
    batches: list[batch_mod.Batch] | None = None,
) -> tuple[list[Path], dict[Path, tuple[str, int, str | None]]]:
    """(every batch's clips dir, clip path -> (batch, item index, spec key)).

    Read from each batch's own state.json — out_dirs are per-batch and may
    have been pointed anywhere with --out. The mapping is by the item's
    RECORDED file name, never by sort order: the file/item correlation is
    exactly what an incident may have corrupted.
    """
    dirs: list[Path] = []
    owners: dict[Path, tuple[str, int, str | None]] = {}
    for b in batches if batches is not None else load_batches():
        out = Path(b.out_dir)
        dirs.append(out)
        for item in b.items:
            if item.file:
                owners[(out / item.file).resolve()] = (
                    b.name,
                    item.index,
                    item_spec_key(item),
                )
    return dirs, owners


def _lost(b: batch_mod.Batch, verified: set[Path], indexes) -> set[int]:
    """Of `indexes`, the items whose file was ACTUALLY deleted.

    State is only edited for a file that is genuinely gone: a deletion can
    be skipped when the bytes no longer match the planned digest (a runner
    replaced the clip), and editing that item anyway would discard fresh,
    correct output.
    """
    return {
        i.index
        for i in b.items
        if i.index in indexes and i.file and (Path(b.out_dir) / i.file).resolve() in verified
    }


def apply_plan(plan: Plan) -> tuple[int, int]:
    """Delete the newer duplicates and reset their items. Returns
    (files deleted, items reset).

    Every deletion is RE-VERIFIED against the digest that justified it:
    the plan holds paths, and a path is not content. A runner regenerating
    an item replaces its clip in place, so an unconditional unlink of a
    planned path can destroy fresh, correct output and reset an item that
    had just succeeded. Each batch is edited under its own claim, so a
    live runner cannot reinstate the repair from its in-memory copy.
    """
    cache = DigestCache.load()
    # Group the work by batch FIRST, and take each batch's claim BEFORE
    # touching its files. Deleting up front and claiming afterwards meant a
    # BatchBusy (a live runner — exactly the case the claim exists for)
    # aborted with the files already gone and zero items reset: `job_id`
    # intact, so the next resume re-adopted the misattributed job this
    # repair existed to escape. Loading state before the lock had the same
    # shape — a runner's newer state was overwritten by a stale snapshot.
    drops_by_batch: dict[str, list[tuple[Path, str]]] = {}
    unowned: list[tuple[Path, str]] = []
    owner_of = {Path(path).resolve(): name for name, _index, path in plan.resets}
    owner_of.update({Path(path).resolve(): name for name, _index, path, _k in plan.skips})
    for group in plan.groups:
        for p in group.drop:
            owner = owner_of.get(p.resolve())
            (drops_by_batch.setdefault(owner, []) if owner else unowned).append((p, group.digest))
    resets_by_batch: dict[str, set[int]] = {}
    for name, index, _path in plan.resets:
        resets_by_batch.setdefault(name, set()).add(index)
    # index -> the surviving file this item's output now lives in.
    skips_by_batch: dict[str, dict[int, str]] = {}
    for name, index, _path, keeper in plan.skips:
        skips_by_batch.setdefault(name, {})[index] = keeper

    deleted = 0

    def _delete(items: list[tuple[Path, str]]) -> set[Path]:
        """Unlink only files whose bytes still match the planned digest."""
        gone: set[Path] = set()
        nonlocal deleted
        for p, digest in items:
            try:
                if cache.digest(p) != digest:
                    continue  # the bytes changed: no longer the planned file
                p.unlink()
            except OSError:
                continue
            deleted += 1
            gone.add(p.resolve())
        return gone

    # Files nobody's state points at: safe to remove with no claim.
    _delete(unowned)

    reset = 0
    for name in sorted(set(drops_by_batch) | set(resets_by_batch) | set(skips_by_batch)):
        b = batch_mod.Batch.load(name)
        if b is None:
            continue
        with batch_mod.claim(b):
            # Re-read UNDER the lock: a runner may have saved since.
            b = batch_mod.Batch.load(name) or b
            verified = _delete(drops_by_batch.get(name, []))
            skipped_lost = _lost(b, verified, skips_by_batch.get(name, {}))
            for item in b.items:
                kept_path = skips_by_batch.get(name, {}).get(item.index)
                if kept_path is None or item.index not in skipped_lost:
                    continue
                # SKIPPED, never PENDING. Its request already has output —
                # the file we just kept — so regenerating would recreate
                # bytes we hold. PENDING here IS the delete/regenerate loop.
                batch_mod.journal(
                    b,
                    "dedupe-skip",
                    item=item.index,
                    prev_job_id=item.job_id,
                    prev_file=item.file,
                    kept=kept_path,
                )
                item.status = batch_mod.SKIPPED
                item.file = None
                item.job_id = None
                item.job_prompt = None
                item.started_at = None
                item.error = f"duplicate of {kept_path} (identical bytes, identical request)"
                reset += 1

            wanted = _lost(b, verified, resets_by_batch.get(name, set()))
            for item in b.items:
                if item.index not in wanted:
                    continue
                # The identity being destroyed goes to the journal FIRST:
                # clearing job_id from state.json is the last record of
                # which server job produced this file, and that is exactly
                # what the next incident will ask.
                batch_mod.journal(
                    b,
                    "dedupe-reset",
                    item=item.index,
                    prev_job_id=item.job_id,
                    prev_file=item.file,
                )
                # PENDING with a CLEAN identity: keeping job_id would make
                # resume re-adopt the wrong job this repair just removed.
                item.status = batch_mod.PENDING
                item.file = None
                item.job_id = None
                item.error = None
                item.started_at = None
                reset += 1
            b.save()
    return deleted, reset
