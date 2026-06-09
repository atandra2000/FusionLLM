---
source_file: "kernels/linear_relu2.py"
type: "rationale"
community: "Linear ReLU²"
location: "L46"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Linear_ReLU
---

# Linear + ReLU²  (pure-PyTorch fallback).      ``out = relu(x @ W.T + bias) ** 2`

## Connections
- [[linear_relu2()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Linear_ReLU