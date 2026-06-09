---
type: community
cohesion: 0.27
members: 10
---

# Shard Index

**Cohesion:** 0.27 - loosely connected
**Members:** 10 nodes

## Members
- [[.__init__()]] - code - data/async_loader.py
- [[.__iter__()]] - code - data/async_loader.py
- [[.__len__()]] - code - data/async_loader.py
- [[.epoch_order()]] - code - data/async_loader.py
- [[.set_shards()]] - code - data/async_loader.py
- [[A view over a list of shards with rank-aware offsets.      Iterating with ``__it]] - rationale - data/async_loader.py
- [[Replace the shard list (used by curriculum hot-swap).]] - rationale - data/async_loader.py
- [[Return the indices that this rank will iterate this epoch.]] - rationale - data/async_loader.py
- [[ShardIndex]] - code - data/async_loader.py
- [[ShardMeta]] - code - data/async_loader.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Shard_Index
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Async Sharding]]
- 3 edges to [[_COMMUNITY_Curriculum Shards]]
- 2 edges to [[_COMMUNITY_Curriculum]]
- 1 edge to [[_COMMUNITY_Async Data Loader]]
- 1 edge to [[_COMMUNITY_Shard Sampler]]

## Top bridge nodes
- [[ShardMeta]] - degree 12, connects to 5 communities
- [[ShardIndex]] - degree 8, connects to 1 community