"""Pytest fixtures for Reddit Reading YouTube Shorts tests."""

import pytest
from src.config import ensure_directories


@pytest.fixture(autouse=True)
def setup_test_directories():
    """Ensure standard project directories are created before each test."""
    ensure_directories()
    yield
