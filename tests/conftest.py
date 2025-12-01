import pytest

from op_tester import bootstrap


@pytest.fixture(scope="session", autouse=True)
def setup_op_tester_registry() -> None:
    """Bootstrap built-in operators/backends once for the entire test session."""

    bootstrap()
