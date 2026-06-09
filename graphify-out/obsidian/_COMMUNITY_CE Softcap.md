---
type: community
cohesion: 0.20
members: 14
---

# CE Softcap

**Cohesion:** 0.20 - loosely connected
**Members:** 14 nodes

## Members
- [[Apply the tanh-based logit softcap in-place.      ``out = softcap_value  tanh(l]] - rationale - kernels/ce_softcap.py
- [[Cross-entropy loss with logit softcap (pure-PyTorch fallback).      Applies ``so]] - rationale - kernels/ce_softcap.py
- [[Fused CE + softcap (pure-PyTorch fallback).]] - rationale - kernels/ce_softcap.py
- [[Fused CE + softcap via Triton (fast path).          Falls back to the pure-PyTor]] - rationale - kernels/ce_softcap.py
- [[Tensor_5]] - code - kernels/ce_softcap.py
- [[Triton forward kernel for fused CE + softcap.          Each program processes on]] - rationale - kernels/ce_softcap.py
- [[_ce_softcap_fwd_kernel()]] - code - kernels/ce_softcap.py
- [[_triton_ce_softcap()]] - code - kernels/ce_softcap.py
- [[ce_softcap()]] - code - kernels/ce_softcap.py
- [[ce_softcap.py]] - code - kernels/ce_softcap.py
- [[constexpr]] - code - kernels/ce_softcap.py
- [[fused_ce_softcap()]] - code - kernels/ce_softcap.py
- [[has_triton()]] - code - kernels/ce_softcap.py
- [[softcap()]] - code - kernels/ce_softcap.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/CE_Softcap
SORT file.name ASC
```
