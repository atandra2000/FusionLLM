---
type: community
cohesion: 0.29
members: 7
---

# Health Checks

**Cohesion:** 0.29 - loosely connected
**Members:** 7 nodes

## Members
- [[._alert()]] - code - training/numerical_health.py
- [[.check_activations()]] - code - training/numerical_health.py
- [[.update_gradients()]] - code - training/numerical_health.py
- [[.update_loss()]] - code - training/numerical_health.py
- [[Check activations for NaN or Inf.                  Args             activations]] - rationale - training/numerical_health.py
- [[Update gradient statistics and check for anomalies.                  Args]] - rationale - training/numerical_health.py
- [[Update loss statistics and check for spikes.                  Args]] - rationale - training/numerical_health.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Health_Checks
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Numerical Health]]
- 1 edge to [[_COMMUNITY_Tensor Validation]]
- 1 edge to [[_COMMUNITY_Activation Monitor]]
- 1 edge to [[_COMMUNITY_Runs Logger]]

## Top bridge nodes
- [[.check_activations()]] - degree 4, connects to 2 communities
- [[.update_gradients()]] - degree 4, connects to 2 communities
- [[.update_loss()]] - degree 4, connects to 2 communities
- [[._alert()]] - degree 4, connects to 1 community