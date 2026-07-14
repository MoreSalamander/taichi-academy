import sys
from pathlib import Path

import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import digital_brain as db


@pytest.fixture(autouse=True)
def fresh_sim():
    db.init_sim(arch=ti.cpu)
    db.apply_seed(0)
    yield
