---
source_file: "kernels/flash_attn.py"
type: "rationale"
community: "Flash Attention"
location: "L42"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Flash_Attention
---

# Dispatch to FA3 or pytorch SDPA.      Args:         query:  ``(bsz, n_heads, seq

## Connections
- [[flash_attention()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Flash_Attention