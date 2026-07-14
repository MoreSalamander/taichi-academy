import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import earth_simulator as es


@pytest.fixture(autouse=True)
def fresh_sim():
    es.init_sim(arch=ti.cpu)
    es.apply_seed(3)
    yield
