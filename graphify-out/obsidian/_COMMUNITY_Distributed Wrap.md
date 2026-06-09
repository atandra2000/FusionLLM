---
type: community
cohesion: 0.24
members: 11
---

# Distributed Wrap

**Cohesion:** 0.24 - loosely connected
**Members:** 11 nodes

## Members
- [[All-to-all list of tensors operation.          Args         output_list Pre-al]] - rationale - utils/distributed.py
- [[Apply FSDP2 (``fully_shard``) to ``model``.      The wrapping policy is per-Tra]] - rationale - utils/distributed.py
- [[Configure ``reshard_after_forward`` per FSDP unit.      Keeps parameter shards r]] - rationale - utils/distributed.py
- [[Module_14]] - code - utils/distributed.py
- [[all_to_all()]] - code - utils/distributed.py
- [[barrier()]] - code - utils/distributed.py
- [[configure_reshard()]] - code - utils/distributed.py
- [[distributed.py]] - code - utils/distributed.py
- [[dtype_4]] - code - utils/distributed.py
- [[is_main_process()]] - code - utils/distributed.py
- [[wrap_fsdp2()]] - code - utils/distributed.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Distributed_Wrap
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_Scheduler & Setup]]
- 5 edges to [[_COMMUNITY_Training Pipeline]]
- 4 edges to [[_COMMUNITY_Distributed Comm]]
- 2 edges to [[_COMMUNITY_Checkpoint Loading]]
- 2 edges to [[_COMMUNITY_Numerical Health]]
- 1 edge to [[_COMMUNITY_NorMuon Optimizer]]

## Top bridge nodes
- [[distributed.py]] - degree 18, connects to 5 communities
- [[is_main_process()]] - degree 7, connects to 3 communities
- [[wrap_fsdp2()]] - degree 7, connects to 1 community
- [[configure_reshard()]] - degree 6, connects to 1 community