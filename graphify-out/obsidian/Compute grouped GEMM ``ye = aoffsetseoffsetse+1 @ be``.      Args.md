---
source_file: "ops/triton/grouped_gemm.py"
type: "rationale"
community: "Expert Dispatch"
location: "L170"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Expert_Dispatch
---

# Compute grouped GEMM: ``y[e] = a[offsets[e]:offsets[e+1]] @ b[e]``.      Args:

## Connections
- [[grouped_gemm()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Expert_Dispatch