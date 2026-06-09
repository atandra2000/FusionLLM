---
source_file: "models/transformer.py"
type: "rationale"
community: "Parallel Embedding"
location: "L467"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Parallel_Embedding
---

# Apply the DeepSeek-V3 logit soft-cap: ``cap * tanh(logits / cap)``.      Bounded

## Connections
- [[softcap_15()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Parallel_Embedding