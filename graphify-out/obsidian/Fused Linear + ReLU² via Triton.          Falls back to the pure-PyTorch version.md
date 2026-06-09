---
source_file: "kernels/linear_relu2.py"
type: "rationale"
community: "Linear ReLU²"
location: "L123"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Linear_ReLU
---

# Fused Linear + ReLU² via Triton.          Falls back to the pure-PyTorch version

## Connections
- [[fused_linear_relu2()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Linear_ReLU