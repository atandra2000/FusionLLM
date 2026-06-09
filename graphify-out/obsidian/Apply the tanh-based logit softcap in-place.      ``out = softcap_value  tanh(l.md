---
source_file: "kernels/ce_softcap.py"
type: "rationale"
community: "CE Softcap"
location: "L41"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/CE_Softcap
---

# Apply the tanh-based logit softcap in-place.      ``out = softcap_value * tanh(l

## Connections
- [[softcap()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/CE_Softcap