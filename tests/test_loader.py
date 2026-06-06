"""Unit tests for `data/async_loader.py` (re-exported through
`tests/test_loader.py` for backward-compat with the Phase 0
placeholder).

Phase 1.5 scope: the real async loader tests live in
`tests/test_async_loader.py`.  This file contains the same tests
re-exported under the original test name so any external
documentation that referenced `test_loader.py` still works.
"""

from __future__ import annotations

import pytest

# Re-export the async loader tests under the loader name
from tests.test_async_loader import (  # noqa: F401  (re-exports)
    TestAsyncShardLoaderSync,
    TestLoadManifest,
    TestOpenShard,
    TestShardHeader,
    TestShardIndex,
)


def test_loader_module_importable():
    """Smoke: the async loader module is importable."""
    from data import async_loader  # noqa: F401

    assert hasattr(async_loader, "AsyncShardLoader")
    assert hasattr(async_loader, "ShardMeta")
    assert hasattr(async_loader, "load_manifest")
    assert hasattr(async_loader, "read_shard_header")
    assert hasattr(async_loader, "open_shard")
