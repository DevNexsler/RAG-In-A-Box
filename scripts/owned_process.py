"""Own a CLI child through startup, cancellation and cleanup on the main thread."""
from contextlib import contextmanager
import os
import signal
import subprocess


@contextmanager
def owned_process(command, **kwargs):
    """Defer cancellation until Popen returns, then kill and reap before unwinding."""
    pending = []
    running = False
    process = None

    def interrupted(signum, _frame):
        if running:
            raise SystemExit(128 + signum)
        pending.append(signum)

    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in originals:
            signal.signal(sig, interrupted)
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        running = True
        if pending:
            raise SystemExit(128 + pending.pop(0))
        yield process
    finally:
        running = False
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        finally:
            for sig, handler in originals.items():
                signal.signal(sig, handler)
        if pending:
            raise SystemExit(128 + pending[0])
