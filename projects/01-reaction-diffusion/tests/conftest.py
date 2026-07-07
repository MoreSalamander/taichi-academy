import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

import gray_scott as gs  # noqa: E402


@pytest.fixture(autouse=True)
def taichi_cpu():
    """Fresh CPU runtime + field allocation per test (mirrors the Metal resize path:
    the only way to 'free' fields is to reset the whole runtime with ti.init)."""
    gs.init_sim(arch=ti.cpu)
    yield
