---
source_file: "utils/checkpoint/manager.py"
type: "rationale"
community: "FSDP Checkpoint"
location: "L138"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/FSDP_Checkpoint
---

# Atomically persist model weights, EMA weights, optimiser state, and metadata.

## Connections
- [[.save()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/FSDP_Checkpoint