---
type: community
cohesion: 0.18
members: 20
---

# MLA Attention

**Cohesion:** 0.18 - loosely connected
**Members:** 20 nodes

## Members
- [[.__init__()_7]] - code - models/mla.py
- [[.__init__()_15]] - code - models/rope.py
- [[._build_attn_mask()]] - code - models/mla.py
- [[._ensure_cache()]] - code - models/mla.py
- [[._get_wkv_b()]] - code - models/mla.py
- [[._invalidate_wkv_b_cache()]] - code - models/mla.py
- [[._rebuild_wkv_b_cache()]] - code - models/mla.py
- [[.forward()_2]] - code - models/mla.py
- [[.prefill_cache()]] - code - models/mla.py
- [[.reset_cache()]] - code - models/mla.py
- [[Compose (sliding window + causal) mask, optionally combine with         an exter]] - rationale - models/mla.py
- [[MultiHeadLatentAttention]] - code - models/mla.py
- [[Per-layer rotary embedding table with YaRN scaling and grow-on-demand.      Args]] - rationale - models/rope.py
- [[RotaryEmbedding]] - code - models/rope.py
- [[Tensor_11]] - code - models/mla.py
- [[bench_mla()]] - code - scripts/bench_mla.py
- [[bench_mla.py]] - code - scripts/bench_mla.py
- [[device_1]] - code - models/mla.py
- [[dtype_3]] - code - models/mla.py
- [[main()_6]] - code - scripts/bench_mla.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/MLA_Attention
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Transformer Blocks]]
- 3 edges to [[_COMMUNITY_Flash Attention]]
- 3 edges to [[_COMMUNITY_Mamba Blocks]]
- 3 edges to [[_COMMUNITY_RoPE]]
- 2 edges to [[_COMMUNITY_Parallel Embedding]]
- 1 edge to [[_COMMUNITY_Compile Benchmarks]]

## Top bridge nodes
- [[MultiHeadLatentAttention]] - degree 24, connects to 5 communities
- [[RotaryEmbedding]] - degree 11, connects to 2 communities
- [[.forward()_2]] - degree 6, connects to 1 community