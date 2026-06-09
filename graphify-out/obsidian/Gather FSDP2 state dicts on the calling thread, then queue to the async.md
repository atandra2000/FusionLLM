---
source_file: "utils/checkpoint/manager.py"
type: "rationale"
community: "FSDP Checkpoint"
location: "L409"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/FSDP_Checkpoint
---

# Gather FSDP2 state dicts on the calling thread, then queue to the async

## Connections
- [[.save_fsdp2_dcp()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/FSDP_Checkpoint