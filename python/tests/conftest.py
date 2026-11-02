import numpy as np
import pytest

import kael


@pytest.fixture(autouse=True)
def seed():
    kael.manual_seed(1234)
    np.random.seed(1234)
