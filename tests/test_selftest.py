"""Smoke tests for the packaged VPSC-GUI application."""

from pathlib import Path
import subprocess
import sys


def test_module_selftest_runs():
    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "FCC_rolling"
    cmd = [sys.executable, "-m", "vpsc_gui", "--self-test", str(example)]
    result = subprocess.run(
        cmd,
        cwd=root,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "repo" / "src")},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "self-test passed" in result.stdout
