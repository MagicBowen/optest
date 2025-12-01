import pytest

from optest import bootstrap


@pytest.fixture(scope="session", autouse=True)
def setup_optest_registry() -> None:
    """Bootstrap built-in operators/backends once for the entire test session."""

    bootstrap()
