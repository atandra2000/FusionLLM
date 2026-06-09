---
type: community
cohesion: 0.20
members: 16
---

# FSDP Checkpoint

**Cohesion:** 0.20 - loosely connected
**Members:** 16 nodes

## Members
- [[._execute_save_fsdp2_dcp()]] - code - utils/checkpoint/manager.py
- [[.keep_last_n()]] - code - utils/checkpoint/manager.py
- [[.load()]] - code - utils/checkpoint/manager.py
- [[.load_fsdp2_dcp()]] - code - utils/checkpoint/manager.py
- [[.save()]] - code - utils/checkpoint/manager.py
- [[.save_async()]] - code - utils/checkpoint/manager.py
- [[.save_fsdp2_dcp()]] - code - utils/checkpoint/manager.py
- [[Atomically persist model weights, EMA weights, optimiser state, and metadata.]] - rationale - utils/checkpoint/manager.py
- [[Delete all but the `n` most recent complete checkpoints.]] - rationale - utils/checkpoint/manager.py
- [[Gather FSDP2 state dicts on the calling thread, then queue to the async]] - rationale - utils/checkpoint/manager.py
- [[Internal DCP save — runs on main or async worker thread.]] - rationale - utils/checkpoint/manager.py
- [[Load FSDP2 model + optimizer state from a DCP checkpoint.]] - rationale - utils/checkpoint/manager.py
- [[Load model weights and optionally restore optimiser state.          Returns meta]] - rationale - utils/checkpoint/manager.py
- [[Module_12]] - code - utils/checkpoint/manager.py
- [[Optimizer_4]] - code - utils/checkpoint/manager.py
- [[Save checkpoint asynchronously (backward compat).]] - rationale - utils/checkpoint/manager.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/FSDP_Checkpoint
SORT file.name ASC
```

## Connections to other communities
- 11 edges to [[_COMMUNITY_Atomic Checkpoint]]
- 8 edges to [[_COMMUNITY_Checkpoint Manager]]
- 4 edges to [[_COMMUNITY_Async Checkpoint]]
- 2 edges to [[_COMMUNITY_Best Model Tracker]]

## Top bridge nodes
- [[._execute_save_fsdp2_dcp()]] - degree 12, connects to 4 communities
- [[.save()]] - degree 11, connects to 3 communities
- [[Optimizer_4]] - degree 9, connects to 2 communities
- [[.keep_last_n()]] - degree 6, connects to 2 communities
- [[.load_fsdp2_dcp()]] - degree 6, connects to 2 communities