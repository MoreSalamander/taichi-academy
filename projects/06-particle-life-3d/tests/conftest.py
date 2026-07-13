import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

import particle_life_3d  # noqa: E402


@pytest.fixture(autouse=True)
def taichi_cpu():
    """Fresh CPU runtime + field allocation per test (Metal-safe resize path)."""
    particle_life_3d.init_sim(arch=ti.cpu)
    yield
