---
source_file: "ops/triton/grouped_gemm.py"
type: "rationale"
community: "Grouped GEMM"
location: "L93"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Grouped_GEMM
---

# Compute grouped GEMM: ``c[e] = a[offsets[e]:offsets[e+1]] @ b[e]``.          Gri

## Connections
- [[_grouped_gemm_kernel()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Grouped_GEMM