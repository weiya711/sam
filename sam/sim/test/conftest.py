import pytest
def pytest_addoption(parser):
    parser.addoption("--debug-sim", action="store_true", default=False, help="Emit debug print statements. Use with `-s`")
    parser.addoption("--ssname", action="store", help="Suitesparse name for the end-to-end test")

@pytest.fixture
def debug_sim(request):
    return request.config.getoption("--debug-sim")

@pytest.fixture
def ssname(request):
    return request.config.getoption("--ssname")
