---
type: community
cohesion: 0.27
members: 10
---

# Config Bundle

**Cohesion:** 0.27 - loosely connected
**Members:** 10 nodes

## Members
- [[.__init__()_29]] - code - training/optimization.py
- [[.__init__()_30]] - code - training/optimization.py
- [[.get_lr()]] - code - training/optimization.py
- [[Composite configuration accepted by class`Pretrainer`.]] - rationale - training/configs.py
- [[ConfigBundle_1]] - code - training/configs.py
- [[ConfigBundle_6]] - code - training/trainer.py
- [[Optimizer_3]] - code - training/optimization.py
- [[Simple warmup + cosine decay scheduler.]] - rationale - training/optimization.py
- [[Tensor_25]] - code - training/trainer.py
- [[WarmupCosineDecayScheduler]] - code - training/optimization.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Config_Bundle
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_Training Pipeline]]
- 11 edges to [[_COMMUNITY_Numerical Health]]
- 10 edges to [[_COMMUNITY_Checkpoint Loading]]
- 6 edges to [[_COMMUNITY_Scheduler & Setup]]
- 5 edges to [[_COMMUNITY_NorMuon Optimizer]]
- 4 edges to [[_COMMUNITY_Scheduling]]
- 3 edges to [[_COMMUNITY_Curriculum]]
- 2 edges to [[_COMMUNITY_Async Data Loader]]
- 2 edges to [[_COMMUNITY_Multi-Token Prediction]]
- 2 edges to [[_COMMUNITY_Compile Benchmarks]]
- 1 edge to [[_COMMUNITY_Cautious Optimizer]]

## Top bridge nodes
- [[ConfigBundle_1]] - degree 39, connects to 7 communities
- [[ConfigBundle_6]] - degree 10, connects to 6 communities
- [[Tensor_25]] - degree 10, connects to 6 communities
- [[WarmupCosineDecayScheduler]] - degree 12, connects to 4 communities
- [[Optimizer_3]] - degree 3, connects to 1 community