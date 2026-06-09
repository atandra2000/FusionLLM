---
type: community
cohesion: 0.25
members: 8
---

# Runs Logger

**Cohesion:** 0.25 - loosely connected
**Members:** 8 nodes

## Members
- [[.__init__()_39]] - code - utils/logging.py
- [[.get_activations()]] - code - training/numerical_health.py
- [[Append-only CSV logger that writes eval metrics to ``runs.csv``.      Created on]] - rationale - utils/logging.py
- [[Get current activations.]] - rationale - training/numerical_health.py
- [[Initialize the runs CSV logger.]] - rationale - training/numerical_health.py
- [[RunsCsvLogger]] - code - utils/logging.py
- [[Tensor_22]] - code - training/numerical_health.py
- [[init_runs_csv()]] - code - training/numerical_health.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Runs_Logger
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Health Monitor]]
- 3 edges to [[_COMMUNITY_Activation Monitor]]
- 3 edges to [[_COMMUNITY_Scheduler & Setup]]
- 1 edge to [[_COMMUNITY_Numerical Health]]
- 1 edge to [[_COMMUNITY_Health Checks]]
- 1 edge to [[_COMMUNITY_Logging]]

## Top bridge nodes
- [[RunsCsvLogger]] - degree 11, connects to 5 communities
- [[init_runs_csv()]] - degree 5, connects to 2 communities
- [[.get_activations()]] - degree 3, connects to 1 community
- [[Tensor_22]] - degree 3, connects to 1 community