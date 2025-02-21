import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dl-data",
        action="store_true",
        help="Run the download tests. This requires an internet connection and can take a long time.",
    )


@pytest.fixture
def dl_data(request):
    return request.config.getoption("--dl-data")
