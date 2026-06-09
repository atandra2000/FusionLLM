---
type: community
cohesion: 0.13
members: 21
---

# Transformer Blocks

**Cohesion:** 0.13 - loosely connected
**Members:** 21 nodes

## Members
- [[.__init__()_5]] - code - models/gated_deltanet.py
- [[.__init__()_20]] - code - models/transformer.py
- [[.__init__()_17]] - code - models/transformer.py
- [[.__init__()_19]] - code - models/transformer.py
- [[.__init__()_18]] - code - models/transformer.py
- [[._get_checkpoint_policy()]] - code - models/transformer.py
- [[.forward()_15]] - code - models/transformer.py
- [[.forward()_12]] - code - models/transformer.py
- [[.forward()_13]] - code - models/transformer.py
- [[.moe_layers()_1]] - code - models/transformer.py
- [[.moe_layers()]] - code - models/transformer.py
- [[A small dense FFN, used as the FFN for Mamba-2 (SSM) layers.      Activation is]] - rationale - models/transformer.py
- [[AsymmetricRescale]] - code - models/transformer.py
- [[DeepSeekMoE_2]] - code - models/transformer.py
- [[DenseFFN]] - code - models/transformer.py
- [[GatedDeltaNet]] - code - models/gated_deltanet.py
- [[Get layer-type-aware activation checkpointing policy.                  Checkpoin]] - rationale - models/transformer.py
- [[One GDN block.  Drop-in for the attention slot in a layer.      Config keys (all]] - rationale - models/gated_deltanet.py
- [[One block.  Slot is either MLA + MoE, or SSM + dense FFN.      ``ssm_type`` (con]] - rationale - models/transformer.py
- [[Per-(channel, token) learnable rescale ``(x - μ)  (σ + ε)  s + b``.      Zero]] - rationale - models/transformer.py
- [[TransformerBlock]] - code - models/transformer.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Transformer_Blocks
SORT file.name ASC
```

## Connections to other communities
- 11 edges to [[_COMMUNITY_Mamba Blocks]]
- 7 edges to [[_COMMUNITY_Parallel Embedding]]
- 5 edges to [[_COMMUNITY_MLA Attention]]
- 5 edges to [[_COMMUNITY_MoLE Model]]
- 4 edges to [[_COMMUNITY_Compile Benchmarks]]
- 3 edges to [[_COMMUNITY_Delta Rule]]
- 1 edge to [[_COMMUNITY_muP]]

## Top bridge nodes
- [[GatedDeltaNet]] - degree 15, connects to 4 communities
- [[.__init__()_19]] - degree 8, connects to 4 communities
- [[TransformerBlock]] - degree 10, connects to 3 communities
- [[AsymmetricRescale]] - degree 9, connects to 3 communities
- [[DenseFFN]] - degree 9, connects to 3 communities