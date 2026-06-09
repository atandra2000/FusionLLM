---
type: community
cohesion: 0.22
members: 11
---

# Linear ReLU²

**Cohesion:** 0.22 - loosely connected
**Members:** 11 nodes

## Members
- [[Fused Linear + ReLU² (pure-PyTorch fallback).]] - rationale - kernels/linear_relu2.py
- [[Fused Linear + ReLU² via Triton.          Falls back to the pure-PyTorch version]] - rationale - kernels/linear_relu2.py
- [[Linear + ReLU²  (pure-PyTorch fallback).      ``out = relu(x @ W.T + bias)  2`]] - rationale - kernels/linear_relu2.py
- [[Tensor_8]] - code - kernels/linear_relu2.py
- [[Triton forward kernel for fused Linear + ReLU².          Each program processes]] - rationale - kernels/linear_relu2.py
- [[_linear_relu2_fwd_kernel()]] - code - kernels/linear_relu2.py
- [[constexpr_2]] - code - kernels/linear_relu2.py
- [[fused_linear_relu2()]] - code - kernels/linear_relu2.py
- [[has_triton()_2]] - code - kernels/linear_relu2.py
- [[linear_relu2()]] - code - kernels/linear_relu2.py
- [[linear_relu2.py]] - code - kernels/linear_relu2.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Linear_ReLU
SORT file.name ASC
```
