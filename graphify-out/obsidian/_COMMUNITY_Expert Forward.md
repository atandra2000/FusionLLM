---
type: community
cohesion: 0.20
members: 11
---

# Expert Forward

**Cohesion:** 0.20 - loosely connected
**Members:** 11 nodes

## Members
- [[.__init__()_8]] - code - models/moe/experts.py
- [[.__init__()_9]] - code - models/moe/moe.py
- [[._refresh_weight_stacks()]] - code - models/moe/moe.py
- [[.forward()_3]] - code - models/moe/experts.py
- [[Expert]] - code - models/moe/experts.py
- [[Refresh precomputed weight stacks after optimizer step.]] - rationale - models/moe/moe.py
- [[Single expert FFN.      Activation is configurable per-instance      ``swiglu]] - rationale - models/moe/experts.py
- [[Single expert forward pass using raw weight tensors (SwiGLU or ReLU²).]] - rationale - models/moe/experts.py
- [[Tensor_13]] - code - models/moe/experts.py
- [[expert_forward_single()]] - code - models/moe/experts.py
- [[experts.py]] - code - models/moe/experts.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Expert_Forward
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Expert Dispatch]]
- 5 edges to [[_COMMUNITY_DeepSeek MoE]]
- 1 edge to [[_COMMUNITY_Routing Gate]]

## Top bridge nodes
- [[Expert]] - degree 9, connects to 2 communities
- [[expert_forward_single()]] - degree 6, connects to 2 communities
- [[.__init__()_9]] - degree 4, connects to 2 communities
- [[experts.py]] - degree 3, connects to 1 community
- [[._refresh_weight_stacks()]] - degree 3, connects to 1 community