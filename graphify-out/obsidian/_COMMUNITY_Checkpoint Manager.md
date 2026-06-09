---
type: community
cohesion: 0.12
members: 18
---

# Checkpoint Manager

**Cohesion:** 0.12 - loosely connected
**Members:** 18 nodes

## Members
- [[.__del__()_1]] - code - utils/checkpoint/manager.py
- [[._checkpoint_complete()]] - code - utils/checkpoint/manager.py
- [[._stop_async_worker()]] - code - utils/checkpoint/manager.py
- [[.delete_checkpoint()]] - code - utils/checkpoint/manager.py
- [[.latest_step()]] - code - utils/checkpoint/manager.py
- [[.list_checkpoints()]] - code - utils/checkpoint/manager.py
- [[.load_weights()]] - code - utils/checkpoint/manager.py
- [[Check if checkpoint for step is complete (backward compat).]] - rationale - utils/checkpoint/manager.py
- [[CheckpointManager_1]] - code - utils/checkpoint/manager.py
- [[Cleanup async worker on object deletion.]] - rationale - utils/checkpoint/manager.py
- [[Load the raw model weights and metadata without applying them.         Used by t]] - rationale - utils/checkpoint/manager.py
- [[Remove all files for a given checkpoint step.]] - rationale - utils/checkpoint/manager.py
- [[Return all complete checkpoint step numbers, sorted ascending.]] - rationale - utils/checkpoint/manager.py
- [[Return the highest complete step number, or None.]] - rationale - utils/checkpoint/manager.py
- [[Save and load model checkpoints.      Features     --------     • Atomic writes]] - rationale - utils/checkpoint/manager.py
- [[Stop the async worker thread (backward compat).]] - rationale - utils/checkpoint/manager.py
- [[__init__.py_10]] - code - utils/checkpoint/__init__.py
- [[manager.py]] - code - utils/checkpoint/manager.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Checkpoint_Manager
SORT file.name ASC
```

## Connections to other communities
- 8 edges to [[_COMMUNITY_FSDP Checkpoint]]
- 6 edges to [[_COMMUNITY_Atomic Checkpoint]]
- 5 edges to [[_COMMUNITY_Async Checkpoint]]
- 3 edges to [[_COMMUNITY_Scheduler & Setup]]
- 3 edges to [[_COMMUNITY_Best Model Tracker]]
- 1 edge to [[_COMMUNITY_Checkpoint Loading]]
- 1 edge to [[_COMMUNITY_Checkpoint Retention]]

## Top bridge nodes
- [[CheckpointManager_1]] - degree 26, connects to 5 communities
- [[manager.py]] - degree 11, connects to 4 communities
- [[__init__.py_10]] - degree 3, connects to 1 community
- [[.load_weights()]] - degree 3, connects to 1 community