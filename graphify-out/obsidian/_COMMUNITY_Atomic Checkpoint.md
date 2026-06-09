---
type: community
cohesion: 0.18
members: 16
---

# Atomic Checkpoint

**Cohesion:** 0.18 - loosely connected
**Members:** 16 nodes

## Members
- [[._save_sync_state()]] - code - utils/checkpoint/manager.py
- [[.save_state_dict()]] - code - utils/checkpoint/manager.py
- [[Atomically persist an already-gathered state dict (used by FSDP path).]] - rationale - utils/checkpoint/manager.py
- [[Build the metadata dict shared by all save paths.          Args         step T]] - rationale - utils/checkpoint/metadata.py
- [[Fallback write a regular (non-DCP) state dict synchronously.]] - rationale - utils/checkpoint/manager.py
- [[JSON serialiser for types that json.dump cannot handle natively.]] - rationale - utils/checkpoint/atomic.py
- [[Path_4]] - code - utils/checkpoint/atomic.py
- [[Pickle an object via torch.save atomically via temp+rename.          Args]] - rationale - utils/checkpoint/atomic.py
- [[Write a JSON file atomically via temp+rename.          Args         obj Dictio]] - rationale - utils/checkpoint/atomic.py
- [[Write a state dict as safetensors atomically via temp+rename.          Args]] - rationale - utils/checkpoint/atomic.py
- [[_json_default()]] - code - utils/checkpoint/atomic.py
- [[atomic.py]] - code - utils/checkpoint/atomic.py
- [[atomic_save_json()]] - code - utils/checkpoint/atomic.py
- [[atomic_save_safetensors()]] - code - utils/checkpoint/atomic.py
- [[atomic_save_torch()]] - code - utils/checkpoint/atomic.py
- [[build_meta()]] - code - utils/checkpoint/metadata.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Atomic_Checkpoint
SORT file.name ASC
```

## Connections to other communities
- 11 edges to [[_COMMUNITY_FSDP Checkpoint]]
- 7 edges to [[_COMMUNITY_Best Model Tracker]]
- 6 edges to [[_COMMUNITY_Checkpoint Manager]]

## Top bridge nodes
- [[atomic_save_json()]] - degree 10, connects to 3 communities
- [[._save_sync_state()]] - degree 10, connects to 3 communities
- [[atomic_save_safetensors()]] - degree 9, connects to 3 communities
- [[.save_state_dict()]] - degree 9, connects to 3 communities
- [[build_meta()]] - degree 7, connects to 3 communities