import pytest
import torch

from scripts.make_demo_data import make_demo_data


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def demo_data(tmp_path):
    return make_demo_data(tmp_path / "data", count=4, size=(16, 32))
