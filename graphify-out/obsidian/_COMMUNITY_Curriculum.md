---
type: community
cohesion: 0.20
members: 14
---

# Curriculum

**Cohesion:** 0.20 - loosely connected
**Members:** 14 nodes

## Members
- [[.advance()]] - code - data/curriculum.py
- [[.stats()_1]] - code - data/curriculum.py
- [[Advance curriculum if needed.      Args         curriculum Curriculum instance]] - rationale - training/curriculum_manager.py
- [[ConfigBundle_2]] - code - training/curriculum_manager.py
- [[Curriculum_1]] - code - training/curriculum_manager.py
- [[Curriculum]] - code - data/curriculum.py
- [[Curriculum manifest and 2-stage sampler.  The pre-training corpus is split into]] - rationale - data/curriculum.py
- [[Hot-swap to stage 2 if ``step = switch_step`` and not already done.          Re]] - rationale - data/curriculum.py
- [[Initialize curriculum learning if configured.      Args         cfg ConfigBund]] - rationale - training/curriculum_manager.py
- [[Two-stage curriculum over the sharded corpus.      Args         manifest_path]] - rationale - data/curriculum.py
- [[advance_curriculum()]] - code - training/curriculum_manager.py
- [[curriculum.py]] - code - data/curriculum.py
- [[curriculum_manager.py]] - code - training/curriculum_manager.py
- [[init_curriculum()]] - code - training/curriculum_manager.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Curriculum
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Curriculum Shards]]
- 4 edges to [[_COMMUNITY_Scheduler & Setup]]
- 3 edges to [[_COMMUNITY_Config Bundle]]
- 2 edges to [[_COMMUNITY_Async Sharding]]
- 2 edges to [[_COMMUNITY_Shard Index]]
- 2 edges to [[_COMMUNITY_Shard Sampler]]
- 2 edges to [[_COMMUNITY_Training Pipeline]]

## Top bridge nodes
- [[Curriculum]] - degree 13, connects to 3 communities
- [[curriculum.py]] - degree 8, connects to 3 communities
- [[curriculum_manager.py]] - degree 7, connects to 3 communities
- [[advance_curriculum()]] - degree 5, connects to 2 communities
- [[init_curriculum()]] - degree 7, connects to 1 community