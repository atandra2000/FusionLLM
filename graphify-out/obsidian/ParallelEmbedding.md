---
source_file: "models/transformer.py"
type: "code"
community: "Parallel Embedding"
location: "L46"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Parallel_Embedding
---

# ParallelEmbedding

## Connections
- [[.__init__()_16]] - `method` [EXTRACTED]
- [[.__init__()_19]] - `calls` [EXTRACTED]
- [[.forward()_11]] - `method` [EXTRACTED]
- [[GatedDeltaNet]] - `uses` [INFERRED]
- [[Mamba2Block]] - `uses` [INFERRED]
- [[MoLE]] - `uses` [INFERRED]
- [[MultiHeadLatentAttention]] - `uses` [INFERRED]
- [[Vocab-sharded embedding. All-reduce on forward; pure embedding lookup when world]] - `rationale_for` [EXTRACTED]
- [[transformer.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Parallel_Embedding