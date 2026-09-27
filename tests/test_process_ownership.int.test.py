"""Cancellation during Popen must not orphan a child before assignment returns."""
import os
import signal
import subprocess
import sys

import pytest

from scripts.owned_process import owned_process


@pytest.mark.parametrize('interrupt', [signal.SIGINT, signal.SIGTERM])
def test_signal_during_process_creation_reaps_child(monkeypatch, interrupt):
    real_popen = subprocess.Popen
    children = []
    def interrupt_after_fork(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        os.kill(os.getpid(), interrupt)
        return child
    monkeypatch.setattr(subprocess, 'Popen', interrupt_after_fork)
    try:
        with pytest.raises(SystemExit) as error:
            with owned_process([sys.executable, '-c', 'import time; time.sleep(60)']):
                pytest.fail('cancelled startup cannot yield a running process')
        assert error.value.code == 128 + interrupt
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
