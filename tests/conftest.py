import pytest

from loadtest.clock import FakeClock


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
