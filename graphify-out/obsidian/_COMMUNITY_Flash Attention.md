---
type: community
cohesion: 0.27
members: 10
---

# Flash Attention

**Cohesion:** 0.27 - loosely connected
**Members:** 10 nodes

## Members
- [[Build attention masks for a long-short window schedule.      In a (period - 1)1]] - rationale - kernels/flash_attn.py
- [[Dispatch to FA3 or pytorch SDPA.      Args         query  ``(bsz, n_heads, seq]] - rationale - kernels/flash_attn.py
- [[Return True if the ``flash_attn`` package is installed and CUDA is available.]] - rationale - kernels/flash_attn.py
- [[Tensor_7]] - code - kernels/flash_attn.py
- [[device]] - code - kernels/flash_attn.py
- [[flash_attention()]] - code - kernels/flash_attn.py
- [[flash_attn.py]] - code - kernels/flash_attn.py
- [[has_flash_attn()]] - code - kernels/flash_attn.py
- [[long_short_window_mask()]] - code - kernels/flash_attn.py
- [[mla.py]] - code - models/mla.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Flash_Attention
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_MLA Attention]]
- 2 edges to [[_COMMUNITY_Mamba Blocks]]
- 1 edge to [[_COMMUNITY_RoPE]]

## Top bridge nodes
- [[mla.py]] - degree 7, connects to 3 communities
- [[flash_attention()]] - degree 6, connects to 1 community