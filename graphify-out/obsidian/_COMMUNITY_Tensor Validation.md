---
type: community
cohesion: 0.20
members: 11
---

# Tensor Validation

**Cohesion:** 0.20 - loosely connected
**Members:** 11 nodes

## Members
- [[Check a loss tensor for NaNInf before backward.      Accepts a torch.Tensor (un]] - rationale - utils/tensor_checks.py
- [[Check a scalar for NaNInf and raise RuntimeError if found.      Used for loss v]] - rationale - utils/tensor_checks.py
- [[Check a tensor for NaNInf and raise RuntimeError if found.      Used for gradie]] - rationale - utils/tensor_checks.py
- [[Check all gradients in a model for NaNInf.      Called before the optimizer ste]] - rationale - utils/tensor_checks.py
- [[Module_17]] - code - utils/tensor_checks.py
- [[Tensor_29]] - code - utils/tensor_checks.py
- [[tensor_checks.py]] - code - utils/tensor_checks.py
- [[validate_gradients()]] - code - utils/tensor_checks.py
- [[validate_loss()]] - code - utils/tensor_checks.py
- [[validate_scalar()]] - code - utils/tensor_checks.py
- [[validate_tensor()]] - code - utils/tensor_checks.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Tensor_Validation
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Health Monitor]]
- 3 edges to [[_COMMUNITY_Numerical Health]]
- 1 edge to [[_COMMUNITY_Health Checks]]

## Top bridge nodes
- [[tensor_checks.py]] - degree 6, connects to 2 communities
- [[validate_scalar()]] - degree 4, connects to 2 communities
- [[validate_loss()]] - degree 5, connects to 1 community
- [[validate_gradients()]] - degree 4, connects to 1 community