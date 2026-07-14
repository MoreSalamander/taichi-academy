import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import voxel_sandbox as vs


@pytest.fixture(autouse=True)
def fresh_sim():
    vs.init_sim(arch=ti.cpu)
    vs.apply_seed()
    yield
