"""Data package — see submodules for the actual surface.

* ``data.dedup``        — MinHash / prefix / MD5 deduplication (Phase 1.1)
* ``data.shard_writer`` — webdataset-style sharded mmap writer (Phase 1.3)
* ``data.async_loader`` — two-stage async shard loader (Phase 1.5)
* ``data.curriculum``   — 2-stage curriculum manifest + sampler (Phase 1.6)
* ``data.prepare_data`` — CLI: collect → dedup → pack → write (Phase 1.2 + 1.4)

Imports are lazy so unit tests for submodules don't require
``datasets`` / ``transformers`` to be installed.
"""

from __future__ import annotations


def __getattr__(name: str):
    """PEP 562 lazy attribute access — keeps top-level imports cheap."""
    if name == "main":
        from data.prepare_data import main

        return main
    raise AttributeError(f"module 'data' has no attribute {name!r}")
