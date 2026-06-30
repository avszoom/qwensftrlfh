"""Code-execution reward: run candidate code against tests in an isolated process.

This is the heart of RLVR (RL from Verifiable Rewards): the reward is not a
learned preference, it is the ground-truth signal "did the code pass the tests?".

WARNING: this executes model-generated code. It runs each candidate in a separate
subprocess with a timeout, which is the standard HumanEval/MBPP approach, but you
should only run it on a trusted machine.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run_program(program: str, timeout: int = 10) -> bool:
    """Execute a self-contained Python program. Return True iff it exits 0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def check_humaneval(completion_program: str, test: str, entry_point: str,
                    timeout: int = 10) -> bool:
    """HumanEval-style check: program defines `entry_point`, then run check()."""
    program = f"{completion_program}\n\n{test}\n\ncheck({entry_point})\n"
    return run_program(program, timeout)


def check_mbpp(code: str, test_list: list[str], timeout: int = 10) -> bool:
    """MBPP-style check: code defines the function, then a list of assert tests.

    Strict / all-or-nothing: True only if EVERY assert passes. Used for benchmarking.
    """
    tests = "\n".join(test_list)
    program = f"{code}\n\n{tests}\n"
    return run_program(program, timeout)


def check_mbpp_fraction(code: str, test_list: list[str], timeout: int = 10) -> float:
    """Dense reward: the FRACTION of individual asserts that pass (0.0 - 1.0).

    Each assert is wrapped in try/except and counted independently, so a candidate
    that passes 3 of 4 tests scores 0.75 instead of 0. This gives RLVR a learning
    signal even when no attempt is fully correct (avoids all-zero reward groups).
    """
    if not test_list:
        return 0.0
    n = len(test_list)
    blocks = []
    for t in test_list:
        blocks.append(f"try:\n    {t.strip()}\n    _p += 1\nexcept Exception:\n    pass")
    program = f"{code}\n_p = 0\n" + "\n".join(blocks) + f"\nprint('FRACPASS', _p, {n})\n"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, timeout=timeout, cwd=tmp, text=True,
            )
        except Exception:
            return 0.0
        for line in proc.stdout.splitlines():
            if line.startswith("FRACPASS"):
                try:
                    _, p, total = line.split()
                    return int(p) / int(total)
                except Exception:
                    return 0.0
    return 0.0


def extract_code(text: str) -> str:
    """Pull a Python code block out of a chat/markdown response, if present."""
    if "```" not in text:
        return text
    blocks = text.split("```")
    # blocks[1] is the first fenced section; strip an optional language tag
    if len(blocks) >= 2:
        body = blocks[1]
        if body.startswith("python"):
            body = body[len("python"):]
        elif body.startswith("py"):
            body = body[len("py"):]
        return body.strip()
    return text
