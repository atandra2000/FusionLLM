---
type: community
cohesion: 0.23
members: 13
---

# Async Sharding

**Cohesion:** 0.23 - loosely connected
**Members:** 13 nodes

## Members
- [[.__init__()_1]] - code - data/async_loader.py
- [[._async_worker_loop()]] - code - data/async_loader.py
- [[.from_bytes()]] - code - data/async_loader.py
- [[CPU worker that fills the pinned-memory buffer.          One micro-batch = one s]] - rationale - data/async_loader.py
- [[Memory-map a shard's data section as a numpy int32 array.      Usage         wi]] - rationale - data/async_loader.py
- [[Path]] - code - data/async_loader.py
- [[Read a ``shardsmanifest.jsonl`` and return the list of rows.]] - rationale - data/async_loader.py
- [[ShardHeader]] - code - data/async_loader.py
- [[Two-stage async sharded loader.  Powers the data path of the pre-training loop.]] - rationale - data/async_loader.py
- [[async_loader.py]] - code - data/async_loader.py
- [[load_manifest()]] - code - data/async_loader.py
- [[open_shard()]] - code - data/async_loader.py
- [[read_shard_header()]] - code - data/async_loader.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Async_Sharding
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Async Data Loader]]
- 4 edges to [[_COMMUNITY_Shard Index]]
- 2 edges to [[_COMMUNITY_Curriculum Shards]]
- 2 edges to [[_COMMUNITY_Curriculum]]
- 1 edge to [[_COMMUNITY_Scheduler & Setup]]

## Top bridge nodes
- [[async_loader.py]] - degree 11, connects to 5 communities
- [[load_manifest()]] - degree 7, connects to 3 communities
- [[.__init__()_1]] - degree 4, connects to 2 communities
- [[open_shard()]] - degree 6, connects to 1 community
- [[._async_worker_loop()]] - degree 3, connects to 1 community