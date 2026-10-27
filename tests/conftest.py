import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--webcam",
        action="store_true",
        default=False,
        help="Run tests that require a physical webcam.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "webcam: mark test as requiring a physical webcam"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--webcam"):
        return
    skip = pytest.mark.skip(reason="Pass --webcam to run tests requiring a camera.")
    for item in items:
        if "webcam" in item.keywords:
            item.add_marker(skip)
