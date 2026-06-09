---
source_file: "models/transformer.py"
type: "rationale"
community: "Transformer Blocks"
location: "L478"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Transformer_Blocks
---

# Per-(channel, token) learnable rescale: ``(x - μ) / (σ + ε) * s + b``.      Zero

## Connections
- [[AsymmetricRescale]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Transformer_Blocks