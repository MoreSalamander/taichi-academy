import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

import galaxy  # noqa: E402


@pytest.fixture(autouse=True)
def taichi_cpu():
    """Fresh CPU runtime + field allocation per test (Metal-safe resize path)."""
    galaxy.init_sim(arch=ti.cpu)
    galaxy.apply_seed(galaxy.seed_galaxy(galaxy.SPIRAL, rng_seed=0))
    yield
