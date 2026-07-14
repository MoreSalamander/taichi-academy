import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import universe_sandbox as us


@pytest.fixture(autouse=True)
def fresh_sim():
    us.init_sim(arch=ti.cpu)
    us.apply_seed("single", seed=1)
    yield
