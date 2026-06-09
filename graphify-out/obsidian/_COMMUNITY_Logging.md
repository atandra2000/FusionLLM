---
type: community
cohesion: 0.15
members: 16
---

# Logging

**Cohesion:** 0.15 - loosely connected
**Members:** 16 nodes

## Members
- [[.__init__()_38]] - code - utils/logging.py
- [[._gpu_stats()]] - code - utils/logging.py
- [[.finish()]] - code - utils/logging.py
- [[.log()_1]] - code - utils/logging.py
- [[.log()]] - code - utils/logging.py
- [[.log_artifact()]] - code - utils/logging.py
- [[.log_moe_routing()]] - code - utils/logging.py
- [[.log_summary()]] - code - utils/logging.py
- [[.log_validation()]] - code - utils/logging.py
- [[.save_log()]] - code - utils/logging.py
- [[Any_2]] - code - utils/logging.py
- [[Log per-expert load histograms (sparse step logging).]] - rationale - utils/logging.py
- [[Logs training and validation metrics to W&B, with stdout as the tertiary sin]] - rationale - utils/logging.py
- [[Tensor_27]] - code - utils/logging.py
- [[TrainerLogger]] - code - utils/logging.py
- [[Upload a file artefact to W&B (rank-0 only).]] - rationale - utils/logging.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Logging
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Scheduler & Setup]]
- 1 edge to [[_COMMUNITY_Runs Logger]]

## Top bridge nodes
- [[TrainerLogger]] - degree 13, connects to 1 community
- [[Any_2]] - degree 3, connects to 1 community
- [[.log()_1]] - degree 2, connects to 1 community