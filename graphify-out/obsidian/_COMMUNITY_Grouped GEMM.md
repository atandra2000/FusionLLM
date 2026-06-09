---
type: community
cohesion: 0.67
members: 3
---

# Grouped GEMM

**Cohesion:** 0.67 - moderately connected
**Members:** 3 nodes

## Members
- [[Compute grouped GEMM ``ce = aoffsetseoffsetse+1 @ be``.          Gri]] - rationale - ops/triton/grouped_gemm.py
- [[_grouped_gemm_kernel()]] - code - ops/triton/grouped_gemm.py
- [[constexpr_3]] - code - ops/triton/grouped_gemm.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Grouped_GEMM
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Expert Dispatch]]

## Top bridge nodes
- [[_grouped_gemm_kernel()]] - degree 3, connects to 1 community