---
type: community
cohesion: 0.31
members: 9
---

# Best Model Tracker

**Cohesion:** 0.31 - loosely connected
**Members:** 9 nodes

## Members
- [[Copy current weights to best.safetensors (and best_ema.safetensors).          Ar]] - rationale - utils/checkpoint/metadata.py
- [[Lock]] - code - utils/checkpoint/metadata.py
- [[Path_6]] - code - utils/checkpoint/metadata.py
- [[Restore best_val_loss from best_meta.json if it exists.          Args         s]] - rationale - utils/checkpoint/metadata.py
- [[Update best checkpoint if val_loss improved (thread-safe).          Args]] - rationale - utils/checkpoint/metadata.py
- [[_update_best()]] - code - utils/checkpoint/metadata.py
- [[load_best_val_loss()]] - code - utils/checkpoint/metadata.py
- [[maybe_update_best()]] - code - utils/checkpoint/metadata.py
- [[metadata.py]] - code - utils/checkpoint/metadata.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Best_Model_Tracker
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_Atomic Checkpoint]]
- 3 edges to [[_COMMUNITY_Checkpoint Manager]]
- 2 edges to [[_COMMUNITY_FSDP Checkpoint]]
- 1 edge to [[_COMMUNITY_Async Checkpoint]]

## Top bridge nodes
- [[maybe_update_best()]] - degree 9, connects to 3 communities
- [[_update_best()]] - degree 8, connects to 3 communities
- [[load_best_val_loss()]] - degree 5, connects to 2 communities
- [[metadata.py]] - degree 6, connects to 1 community