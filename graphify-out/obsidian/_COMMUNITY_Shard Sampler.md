---
type: community
cohesion: 0.40
members: 5
---

# Shard Sampler

**Cohesion:** 0.40 - moderately connected
**Members:** 5 nodes

## Members
- [[.iter_active()]] - code - data/curriculum.py
- [[.sample()_1]] - code - data/curriculum.py
- [[Return a list of shards in the active stage's in-scope sources.          Used]] - rationale - data/curriculum.py
- [[Sample one shard from the active stage.]] - rationale - data/curriculum.py
- [[ShardMeta_1]] - code - data/curriculum.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Shard_Sampler
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Curriculum]]
- 1 edge to [[_COMMUNITY_Shard Index]]
- 1 edge to [[_COMMUNITY_Curriculum Shards]]

## Top bridge nodes
- [[ShardMeta_1]] - degree 4, connects to 2 communities
- [[.iter_active()]] - degree 3, connects to 1 community
- [[.sample()_1]] - degree 3, connects to 1 community