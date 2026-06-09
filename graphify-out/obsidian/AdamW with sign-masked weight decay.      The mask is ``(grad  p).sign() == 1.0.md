---
source_file: "training/optimization.py"
type: "rationale"
community: "Numerical Health"
location: "L119"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Numerical_Health
---

# AdamW with sign-masked weight decay.      The mask is ``(grad * p).sign() == 1.0

## Connections
- [[CautiousAdamW]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Numerical_Health