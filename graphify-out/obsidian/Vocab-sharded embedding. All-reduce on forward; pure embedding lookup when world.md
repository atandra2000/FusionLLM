---
source_file: "models/transformer.py"
type: "rationale"
community: "Parallel Embedding"
location: "L47"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Parallel_Embedding
---

# Vocab-sharded embedding. All-reduce on forward; pure embedding lookup when world

## Connections
- [[ParallelEmbedding]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Parallel_Embedding