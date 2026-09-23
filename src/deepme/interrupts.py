"""One Ctrl-C protocol for the whole CLI — owned here, inherited everywhere.

Per-command interrupt handling was re-implemented per command, so every
command not yet visited still detonated a stack trace — and even the fixed
ones did on a SECOND ^C, which lands during the interpreter's atexit
thread-join (concurrent.futures joins its workers at shutdown; a lingering
poll thread turns that join into a KeyboardInterrupt traceback).

The protocol, installed once in main():

  first ^C   cooperative — CANCEL is set (every generation poll loop watches
             it and stops within one cycle) and KeyboardInterrupt rises so
             the running command prints ITS summary; commands without a
             bespoke handler get main()'s plain "interrupted" instead of a
             traceback.
  second ^C  immediate — os._exit(130). No atexit, no thread joins, no
             traceback. The first ^C already made the state safe (state
             files are written atomically per item; server jobs continue and
             are re-attachable), so there is nothing a forced exit can
             corrupt — only waits it can skip.

Every generation is interrupt-aware with zero wiring: media.generate_video
defaults its cancel event to CANCEL, so story/showcase/batch/generate all
stop within one poll cycle of the first ^C.
"""

from __future__ import annotations

import os
import signal
import threading

# The process-wide cooperative stop signal. Poll loops check it every cycle.
CANCEL = threading.Event()

_hits = {"n": 0}


def install() -> None:
    """Install the two-stage handler. Call once, from the main thread."""

    def handler(signum, frame):  # the signal API's required shape
        _hits["n"] += 1
        if _hits["n"] >= 2:
            os.write(
                2,
                b"\nforce quit (second ^C) - jobs continue "
                b"server-side and re-attach on the next run\n",
            )
            os._exit(130)
        CANCEL.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
