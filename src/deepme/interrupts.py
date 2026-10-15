"""One Ctrl-C protocol for the whole CLI — owned here, inherited everywhere.

Per-command interrupt handling was re-implemented per command, so every
command not yet visited still detonated a stack trace.

The protocol, installed once in main():

  first ^C   cooperative — CANCEL is set (every wait in the generation path
             watches it) and KeyboardInterrupt rises so the running command
             prints ITS summary; commands without a bespoke handler get
             main()'s plain "interrupted" instead of a traceback.
  second ^C  escalated, still clean — SystemExit(130) through the normal
             machinery: finally blocks run, atexit runs, worker threads
             join. THEN THE HANDLER STANDS DOWN (see below).
  third ^C   the OS. SIGINT is restored to its default disposition, so a
             further ^C terminates the process the way the kernel does it.
             We never call os._exit ourselves.

WHY THE HANDLER STANDS DOWN, and why holding ^C used to be the only way
out. After the main thread finishes, the interpreter joins non-daemon
threads inside `threading._shutdown` -> `concurrent.futures.thread.
_python_exit` -> `t.join()`. A signal arriving there runs THIS handler on
the main thread, inside that atexit hook — and an exception raised from an
atexit hook is caught and printed as "Exception ignored on threading
shutdown", after which THE JOIN RESUMES. So every ^C after the first two
was swallowed by the shutdown machinery and did nothing at all; the user
was left holding the key.

Two changes make the join finish instead:

  * every wait a worker performs is cancellable — `Event.wait(n)` rather
    than `time.sleep(n)` in the poll loop, a per-chunk check in the
    download — so workers exit in well under a second once CANCEL is set;
  * the handler removes itself after escalating, so it can never raise
    into shutdown machinery that is going to swallow it.
"""

from __future__ import annotations

import os
import signal
import threading

# The process-wide cooperative stop signal. Every wait in the generation
# path (poll interval, download chunks, retry backoff, worker pickup)
# watches it, which is what makes teardown prompt rather than "eventual".
CANCEL = threading.Event()

_hits = {"n": 0}


def install() -> None:
    """Install the staged handler. Call once, from the main thread.

    Terminal restoration is deliberately NOT handled here: rich's Live and
    Progress restore cursor/colors/wrap when their context managers exit,
    and both KeyboardInterrupt and SystemExit unwind through them. The
    shredded-terminal bug existed only because a hard os._exit bypassed
    that machinery — the fix is not bypassing it, not repainting the
    terminal by hand.
    """

    def handler(signum, frame):  # the signal API's required shape
        _hits["n"] += 1
        CANCEL.set()
        if _hits["n"] >= 2:
            # os.write, not print: the only signal-safe output.
            os.write(
                2,
                b"\nstopping now - jobs continue server-side and re-attach on the next run\n",
            )
            # STAND DOWN before raising. From here the process is unwinding
            # and will soon be inside atexit thread-joins, where a raise
            # from this handler is swallowed and the join simply continues
            # — the reason holding ^C was previously the only way to stop.
            # Restoring the default disposition means any further ^C is
            # handled by the OS instead of disappearing into our own
            # shutdown path.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise SystemExit(130)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
