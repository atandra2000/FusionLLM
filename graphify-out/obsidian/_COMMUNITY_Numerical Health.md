---
type: community
cohesion: 0.15
members: 28
---

# Numerical Health

**Cohesion:** 0.15 - loosely connected
**Members:** 28 nodes

## Members
- [[.__init__()_27]] - code - training/numerical_health.py
- [[.get_stats()]] - code - training/numerical_health.py
- [[.reset()]] - code - training/numerical_health.py
- [[AdamW]] - code
- [[AdamW with sign-masked weight decay.      The mask is ``(grad  p).sign() == 1.0]] - rationale - training/optimization.py
- [[All-reduce a tensor and return its mean across all ranks.]] - rationale - utils/distributed.py
- [[CautiousAdamW_1]] - code - training/train_step.py
- [[CautiousAdamW]] - code - training/optimization.py
- [[Compute the total loss (CE + balance + z-loss).      Args         model The ma]] - rationale - training/train_step.py
- [[ConfigBundle_5]] - code - training/train_step.py
- [[Execute a single optimizer step with gradient clipping.      Args         raw_m]] - rationale - training/train_step.py
- [[Execute a single training step.      Args         model The main model (or MTP]] - rationale - training/train_step.py
- [[Get current statistics.]] - rationale - training/numerical_health.py
- [[GradScaler]] - code - training/train_step.py
- [[Module_10]] - code - training/train_step.py
- [[Monitors numerical health during training.]] - rationale - training/numerical_health.py
- [[Muon_2]] - code - training/train_step.py
- [[NumericalHealthMonitor_1]] - code - training/train_step.py
- [[NumericalHealthMonitor]] - code - training/numerical_health.py
- [[Reset all statistics.]] - rationale - training/numerical_health.py
- [[Tensor_24]] - code - training/train_step.py
- [[_LRScheduler]] - code - training/train_step.py
- [[all_reduce_mean()]] - code - utils/distributed.py
- [[compute_loss()_1]] - code - training/train_step.py
- [[device_4]] - code - training/train_step.py
- [[optimizer_step()]] - code - training/train_step.py
- [[train_step()]] - code - training/train_step.py
- [[train_step.py]] - code - training/train_step.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Numerical_Health
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_Checkpoint Loading]]
- 11 edges to [[_COMMUNITY_Config Bundle]]
- 7 edges to [[_COMMUNITY_Health Monitor]]
- 4 edges to [[_COMMUNITY_Health Checks]]
- 4 edges to [[_COMMUNITY_Scheduler & Setup]]
- 3 edges to [[_COMMUNITY_Tensor Validation]]
- 3 edges to [[_COMMUNITY_Distributed Comm]]
- 2 edges to [[_COMMUNITY_Training Pipeline]]
- 2 edges to [[_COMMUNITY_NorMuon Optimizer]]
- 2 edges to [[_COMMUNITY_Distributed Wrap]]
- 1 edge to [[_COMMUNITY_Runs Logger]]
- 1 edge to [[_COMMUNITY_Cautious Optimizer]]

## Top bridge nodes
- [[train_step.py]] - degree 15, connects to 7 communities
- [[CautiousAdamW]] - degree 18, connects to 5 communities
- [[NumericalHealthMonitor]] - degree 24, connects to 3 communities
- [[all_reduce_mean()]] - degree 8, connects to 3 communities
- [[_LRScheduler]] - degree 7, connects to 3 communities