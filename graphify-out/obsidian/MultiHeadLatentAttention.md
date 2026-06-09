---
source_file: "models/mla.py"
type: "code"
community: "MLA Attention"
location: "L31"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/MLA_Attention
---

# MultiHeadLatentAttention

## Connections
- [[.__init__()_7]] - `method` [EXTRACTED]
- [[.__init__()_18]] - `calls` [EXTRACTED]
- [[._build_attn_mask()]] - `method` [EXTRACTED]
- [[._ensure_cache()]] - `method` [EXTRACTED]
- [[._get_wkv_b()]] - `method` [EXTRACTED]
- [[._invalidate_wkv_b_cache()]] - `method` [EXTRACTED]
- [[._rebuild_wkv_b_cache()]] - `method` [EXTRACTED]
- [[.forward()_2]] - `method` [EXTRACTED]
- [[.prefill_cache()]] - `method` [EXTRACTED]
- [[.reset_cache()]] - `method` [EXTRACTED]
- [[AsymmetricRescale]] - `uses` [INFERRED]
- [[DeepSeekMoE_2]] - `uses` [INFERRED]
- [[DenseFFN]] - `uses` [INFERRED]
- [[Module_5]] - `uses` [INFERRED]
- [[ParallelEmbedding]] - `uses` [INFERRED]
- [[RotaryEmbedding]] - `uses` [INFERRED]
- [[Tensor_19]] - `uses` [INFERRED]
- [[Transformer]] - `uses` [INFERRED]
- [[TransformerBlock]] - `uses` [INFERRED]
- [[__init__.py_4]] - `imports` [EXTRACTED]
- [[bench_mla()]] - `calls` [EXTRACTED]
- [[bench_mla.py]] - `imports` [EXTRACTED]
- [[mla.py]] - `contains` [EXTRACTED]
- [[transformer.py]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/MLA_Attention