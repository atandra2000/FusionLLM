---
type: community
cohesion: 0.21
members: 17
---

# Expert Dispatch

**Cohesion:** 0.21 - loosely connected
**Members:** 17 nodes

## Members
- [[All-to-all expert dispatch (DeepSeek-V3 style) - falls back to scatter-gather.]] - rationale - models/moe/dispatch.py
- [[Compute grouped GEMM ``ye = aoffsetseoffsetse+1 @ be``.      Args]] - rationale - ops/triton/grouped_gemm.py
- [[Return True if Triton is importable and a CUDA device is available.]] - rationale - ops/triton/grouped_gemm.py
- [[Scatter-gather dispatch iterate active experts, compute, scatter-add.]] - rationale - models/moe/dispatch.py
- [[Tensor_12]] - code - models/moe/dispatch.py
- [[Tensor_20]] - code - ops/triton/grouped_gemm.py
- [[Try the Triton grouped-GEMM fast-path.  Returns True on success.]] - rationale - models/moe/dispatch.py
- [[__init__.py_5]] - code - models/moe/__init__.py
- [[_autotune_configs()_1]] - code - ops/triton/grouped_gemm.py
- [[all_to_all_dispatch()]] - code - models/moe/dispatch.py
- [[dispatch.py]] - code - models/moe/dispatch.py
- [[grouped_gemm()]] - code - ops/triton/grouped_gemm.py
- [[grouped_gemm.py]] - code - ops/triton/grouped_gemm.py
- [[has_triton()_3]] - code - ops/triton/grouped_gemm.py
- [[moe.py]] - code - models/moe/moe.py
- [[scatter_gather_dispatch()]] - code - models/moe/dispatch.py
- [[try_grouped_gemm()]] - code - models/moe/dispatch.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Expert_Dispatch
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Expert Forward]]
- 4 edges to [[_COMMUNITY_Routing Gate]]
- 2 edges to [[_COMMUNITY_DeepSeek MoE]]
- 1 edge to [[_COMMUNITY_Mamba Blocks]]
- 1 edge to [[_COMMUNITY_Grouped GEMM]]

## Top bridge nodes
- [[__init__.py_5]] - degree 13, connects to 4 communities
- [[moe.py]] - degree 6, connects to 3 communities
- [[grouped_gemm.py]] - degree 4, connects to 1 community