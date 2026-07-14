import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import destruction as dz


@pytest.fixture(autouse=True)
def fresh_sim():
    dz.init_sim(arch=ti.cpu)
    dz.apply_seed()
    yield
