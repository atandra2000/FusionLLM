---
type: community
cohesion: 0.17
members: 20
---

# Checkpoint Loading

**Cohesion:** 0.17 - loosely connected
**Members:** 20 nodes

## Members
- [[.step()_1]] - code - training/optimization.py
- [[CheckpointManager]] - code - training/checkpointing.py
- [[ConfigBundle]] - code - training/checkpointing.py
- [[Find the latest complete checkpoint step.]] - rationale - training/checkpointing.py
- [[Load a checkpoint.      Args         step Checkpoint step to load         cfg]] - rationale - training/checkpointing.py
- [[Module_6]] - code - training/checkpointing.py
- [[Muon]] - code - training/checkpointing.py
- [[Muon_1]] - code - training/optimization.py
- [[Muon optimizer — Newton-Schulz orthogonalized momentum for matrix     parameters]] - rationale - training/optimization.py
- [[Newton-Schulz orthogonalization for the Muon optimizer.      Approximates U @ V.]] - rationale - training/optimization.py
- [[Optimizer_1]] - code - training/checkpointing.py
- [[Save a checkpoint.      Args         step Current training step         tag C]] - rationale - training/checkpointing.py
- [[Verify NorMuon optimizer state was restored correctly.]] - rationale - training/checkpointing.py
- [[_verify_nor_muon_state()]] - code - training/checkpointing.py
- [[_zeropower_via_newtonschulz5()]] - code - training/optimization.py
- [[checkpointing.py]] - code - training/checkpointing.py
- [[find_latest_checkpoint()]] - code - training/checkpointing.py
- [[load_checkpoint()]] - code - training/checkpointing.py
- [[optimization.py]] - code - training/optimization.py
- [[save_checkpoint()]] - code - training/checkpointing.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Checkpoint_Loading
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_Numerical Health]]
- 10 edges to [[_COMMUNITY_Config Bundle]]
- 6 edges to [[_COMMUNITY_Training Pipeline]]
- 5 edges to [[_COMMUNITY_Scheduler & Setup]]
- 5 edges to [[_COMMUNITY_NorMuon Optimizer]]
- 2 edges to [[_COMMUNITY_Cautious Optimizer]]
- 2 edges to [[_COMMUNITY_Distributed Wrap]]
- 1 edge to [[_COMMUNITY_Checkpoint Manager]]

## Top bridge nodes
- [[optimization.py]] - degree 16, connects to 7 communities
- [[Muon_1]] - degree 24, connects to 4 communities
- [[checkpointing.py]] - degree 10, connects to 4 communities
- [[find_latest_checkpoint()]] - degree 5, connects to 2 communities
- [[_zeropower_via_newtonschulz5()]] - degree 5, connects to 2 communities