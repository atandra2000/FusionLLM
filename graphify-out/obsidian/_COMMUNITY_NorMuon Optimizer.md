---
type: community
cohesion: 0.14
members: 18
---

# NorMuon Optimizer

**Cohesion:** 0.14 - loosely connected
**Members:** 18 nodes

## Members
- [[.__init__()_26]] - code - training/normuon.py
- [[.get_config_summary()]] - code - training/normuon.py
- [[.step()]] - code - training/normuon.py
- [[Build the optimizer pair.      When ``cfg.optimizer == muon_adamw`` (Muon, Ca]] - rationale - training/optimization.py
- [[ConfigBundle_3]] - code - training/optimization.py
- [[Get summary of optimizer configuration.]] - rationale - training/normuon.py
- [[Module_7]] - code - training/normuon.py
- [[Module_9]] - code - training/optimization.py
- [[NorMuon_1]] - code - training/optimization.py
- [[NorMuon]] - code - training/normuon.py
- [[NorMuon — orthogonalized Adam with per-row RMS for matrix params.      Args]] - rationale - training/normuon.py
- [[Optimizer_2]] - code
- [[Validate NorMuon configuration and return warnings.          Args         lr L]] - rationale - training/normuon.py
- [[Validate parameter groups for NorMuon.          Args         param_groups List]] - rationale - training/normuon.py
- [[build_optimizers()]] - code - training/optimization.py
- [[normuon.py]] - code - training/normuon.py
- [[validate_normuon_config()]] - code - training/normuon.py
- [[validate_param_groups()]] - code - training/normuon.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/NorMuon_Optimizer
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Config Bundle]]
- 5 edges to [[_COMMUNITY_Checkpoint Loading]]
- 2 edges to [[_COMMUNITY_Numerical Health]]
- 2 edges to [[_COMMUNITY_Scheduler & Setup]]
- 1 edge to [[_COMMUNITY_Training Pipeline]]
- 1 edge to [[_COMMUNITY_Cautious Optimizer]]
- 1 edge to [[_COMMUNITY_Distributed Wrap]]

## Top bridge nodes
- [[build_optimizers()]] - degree 12, connects to 5 communities
- [[NorMuon]] - degree 16, connects to 4 communities
- [[normuon.py]] - degree 4, connects to 1 community
- [[NorMuon_1]] - degree 3, connects to 1 community
- [[ConfigBundle_3]] - degree 3, connects to 1 community