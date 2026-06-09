---
type: community
cohesion: 0.12
members: 20
---

# Async Checkpoint

**Cohesion:** 0.12 - loosely connected
**Members:** 20 nodes

## Members
- [[.__init__()_35]] - code - utils/checkpoint/async_worker.py
- [[.__init__()_36]] - code - utils/checkpoint/manager.py
- [[._shard_dir()]] - code - utils/checkpoint/manager.py
- [[._start()]] - code - utils/checkpoint/async_worker.py
- [[._step_dir()]] - code - utils/checkpoint/manager.py
- [[._worker_loop()]] - code - utils/checkpoint/async_worker.py
- [[.is_running()]] - code - utils/checkpoint/async_worker.py
- [[.stop()_1]] - code - utils/checkpoint/async_worker.py
- [[.submit()]] - code - utils/checkpoint/async_worker.py
- [[AsyncCheckpointWorker]] - code - utils/checkpoint/async_worker.py
- [[Background thread for async checkpoint operations.]] - rationale - utils/checkpoint/async_worker.py
- [[Background thread that processes async checkpoint requests.]] - rationale - utils/checkpoint/async_worker.py
- [[Check if the worker thread is running.]] - rationale - utils/checkpoint/async_worker.py
- [[Directory for a given step (used in sharded mode).]] - rationale - utils/checkpoint/manager.py
- [[Path_5]] - code - utils/checkpoint/manager.py
- [[Per-rank directory within a step directory.]] - rationale - utils/checkpoint/manager.py
- [[Start background thread.]] - rationale - utils/checkpoint/async_worker.py
- [[Stop background thread and wait for pending operations.]] - rationale - utils/checkpoint/async_worker.py
- [[Submit an operation to the async worker.                  Args             oper]] - rationale - utils/checkpoint/async_worker.py
- [[async_worker.py]] - code - utils/checkpoint/async_worker.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Async_Checkpoint
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Checkpoint Manager]]
- 4 edges to [[_COMMUNITY_FSDP Checkpoint]]
- 1 edge to [[_COMMUNITY_Best Model Tracker]]

## Top bridge nodes
- [[AsyncCheckpointWorker]] - degree 14, connects to 2 communities
- [[._step_dir()]] - degree 6, connects to 2 communities
- [[.__init__()_36]] - degree 4, connects to 2 communities
- [[._shard_dir()]] - degree 4, connects to 1 community