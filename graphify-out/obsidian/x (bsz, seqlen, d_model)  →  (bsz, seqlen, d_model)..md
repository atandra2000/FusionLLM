---
source_file: "models/gated_deltanet.py"
type: "rationale"
community: "Delta Rule"
location: "L116"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Delta_Rule
---

# x: (bsz, seqlen, d_model)  →  (bsz, seqlen, d_model).

## Connections
- [[.forward()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Delta_Rule