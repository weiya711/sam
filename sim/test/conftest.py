import pytest
def pytest_addoption(parser):
    parser.addoption("--debug-sim", action="store_true", default=False, help="Emit debug print statements. Use with `-s`")

@pytest.fixture
def debug_sim(request):
    return request.config.getoption("--debug-sim")