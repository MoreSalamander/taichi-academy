import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

import artificial_life  # noqa: E402


@pytest.fixture(autouse=True)
def taichi_cpu():
    """Fresh CPU runtime + field allocation per test (Metal-safe resize path)."""
    artificial_life.init_sim(arch=ti.cpu)
    artificial_life.apply_seed()
    yield
