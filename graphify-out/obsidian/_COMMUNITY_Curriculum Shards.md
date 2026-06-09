---
type: community
cohesion: 0.18
members: 13
---

# Curriculum Shards

**Cohesion:** 0.18 - loosely connected
**Members:** 13 nodes

## Members
- [[.__init__()_2]] - code - data/curriculum.py
- [[.__len__()_1]] - code - data/curriculum.py
- [[.__post_init__()]] - code - data/curriculum.py
- [[._build_alias()]] - code - data/curriculum.py
- [[.active()]] - code - data/curriculum.py
- [[.in_scope_sources()]] - code - data/curriculum.py
- [[.sample()]] - code - data/curriculum.py
- [[Build the Vose alias table over the per-shard weights.]] - rationale - data/curriculum.py
- [[CurriculumStage]] - code - data/curriculum.py
- [[One stage of the curriculum.      Holds a list of ``ShardMeta`` (all of them) pl]] - rationale - data/curriculum.py
- [[Path_1]] - code - data/curriculum.py
- [[Random]] - code - data/curriculum.py
- [[Sample one shard with replacement, weighted by stage weight.]] - rationale - data/curriculum.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Curriculum_Shards
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Curriculum]]
- 3 edges to [[_COMMUNITY_Shard Index]]
- 2 edges to [[_COMMUNITY_Async Sharding]]
- 1 edge to [[_COMMUNITY_Shard Sampler]]
- 1 edge to [[_COMMUNITY_Data Preparation]]

## Top bridge nodes
- [[Random]] - degree 6, connects to 4 communities
- [[CurriculumStage]] - degree 10, connects to 2 communities
- [[.__init__()_2]] - degree 5, connects to 2 communities
- [[.sample()]] - degree 5, connects to 1 community
- [[.active()]] - degree 2, connects to 1 community