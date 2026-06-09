---
type: community
cohesion: 0.24
members: 10
---

# Parallel Embedding

**Cohesion:** 0.24 - loosely connected
**Members:** 10 nodes

## Members
- [[.__init__()_16]] - code - models/transformer.py
- [[.forward()_11]] - code - models/transformer.py
- [[.forward()_14]] - code - models/transformer.py
- [[.forward_with_hidden()]] - code - models/transformer.py
- [[Apply the DeepSeek-V3 logit soft-cap ``cap  tanh(logits  cap)``.      Bounded]] - rationale - models/transformer.py
- [[Forward that also returns the pre-head hidden state.  Used by MTP.]] - rationale - models/transformer.py
- [[ParallelEmbedding]] - code - models/transformer.py
- [[Tensor_19]] - code - models/transformer.py
- [[Vocab-sharded embedding. All-reduce on forward; pure embedding lookup when world]] - rationale - models/transformer.py
- [[softcap_15()]] - code - models/transformer.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Parallel_Embedding
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_Transformer Blocks]]
- 4 edges to [[_COMMUNITY_Mamba Blocks]]
- 2 edges to [[_COMMUNITY_MLA Attention]]
- 2 edges to [[_COMMUNITY_MoLE Model]]
- 2 edges to [[_COMMUNITY_Compile Benchmarks]]

## Top bridge nodes
- [[Tensor_19]] - degree 11, connects to 4 communities
- [[ParallelEmbedding]] - degree 9, connects to 4 communities
- [[softcap_15()]] - degree 5, connects to 1 community
- [[.forward_with_hidden()]] - degree 4, connects to 1 community
- [[.forward()_14]] - degree 3, connects to 1 community