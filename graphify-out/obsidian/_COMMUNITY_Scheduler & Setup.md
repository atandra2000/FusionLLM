---
type: community
cohesion: 0.24
members: 14
---

# Scheduler & Setup

**Cohesion:** 0.24 - loosely connected
**Members:** 14 nodes

## Members
- [[.__init__()_33]] - code - training/trainer.py
- [[.__init__()_34]] - code - training/wsd.py
- [[.get_lr()_1]] - code - training/wsd.py
- [[Initialize the distributed process group.      Returns         (world_size, ran]] - rationale - utils/distributed.py
- [[Module-level initialiser (called by the trainer once).]] - rationale - utils/logging.py
- [[WSDScheduler]] - code - training/wsd.py
- [[Warmup-Stable-Decay scheduler.      Args         optimizer wrapped optimizer(s]] - rationale - training/wsd.py
- [[__init__.py_9]] - code - utils/__init__.py
- [[get_logger()]] - code - utils/logging.py
- [[init_logging()]] - code - utils/logging.py
- [[logging.py]] - code - utils/logging.py
- [[setup_distributed()]] - code - utils/distributed.py
- [[trainer.py]] - code - training/trainer.py
- [[wsd.py]] - code - training/wsd.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Scheduler__Setup
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_Training Pipeline]]
- 7 edges to [[_COMMUNITY_Distributed Wrap]]
- 6 edges to [[_COMMUNITY_Config Bundle]]
- 5 edges to [[_COMMUNITY_Checkpoint Loading]]
- 5 edges to [[_COMMUNITY_Health Monitor]]
- 5 edges to [[_COMMUNITY_Scheduling]]
- 4 edges to [[_COMMUNITY_Curriculum]]
- 4 edges to [[_COMMUNITY_Numerical Health]]
- 4 edges to [[_COMMUNITY_Logging]]
- 3 edges to [[_COMMUNITY_Multi-Token Prediction]]
- 3 edges to [[_COMMUNITY_Mamba Blocks]]
- 3 edges to [[_COMMUNITY_Runs Logger]]
- 3 edges to [[_COMMUNITY_Checkpoint Manager]]
- 2 edges to [[_COMMUNITY_Compile Benchmarks]]
- 2 edges to [[_COMMUNITY_NorMuon Optimizer]]
- 2 edges to [[_COMMUNITY_Hardware Config]]
- 1 edge to [[_COMMUNITY_Async Sharding]]
- 1 edge to [[_COMMUNITY_Async Data Loader]]

## Top bridge nodes
- [[trainer.py]] - degree 45, connects to 16 communities
- [[.__init__()_33]] - degree 20, connects to 11 communities
- [[__init__.py_9]] - degree 11, connects to 5 communities
- [[WSDScheduler]] - degree 10, connects to 3 communities
- [[logging.py]] - degree 6, connects to 2 communities