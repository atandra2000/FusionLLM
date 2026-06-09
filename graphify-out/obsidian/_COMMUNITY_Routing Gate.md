---
type: community
cohesion: 0.24
members: 11
---

# Routing Gate

**Cohesion:** 0.24 - loosely connected
**Members:** 11 nodes

## Members
- [[.__init__()_10]] - code - models/moe/routing.py
- [[.forward()_5]] - code - models/moe/routing.py
- [[.get_z_loss()_1]] - code - models/moe/routing.py
- [[.update_bias()]] - code - models/moe/routing.py
- [[Args             x (T, dim) — flattened token representations          Returns]] - rationale - models/moe/routing.py
- [[AuxLossFreeGate]] - code - models/moe/routing.py
- [[Auxiliary-Loss-Free Load Balancing Gate (DeepSeek-V3).      Routing decision]] - rationale - models/moe/routing.py
- [[Shared sortsegmentcapacity logic for scatter-gather routing.      Returns]] - rationale - models/moe/routing.py
- [[Tensor_15]] - code - models/moe/routing.py
- [[compute_routing_segments()]] - code - models/moe/routing.py
- [[routing.py]] - code - models/moe/routing.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Routing_Gate
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Expert Dispatch]]
- 2 edges to [[_COMMUNITY_MoE Routing]]
- 2 edges to [[_COMMUNITY_DeepSeek MoE]]
- 1 edge to [[_COMMUNITY_Mamba Blocks]]
- 1 edge to [[_COMMUNITY_Expert Forward]]

## Top bridge nodes
- [[AuxLossFreeGate]] - degree 14, connects to 5 communities
- [[compute_routing_segments()]] - degree 4, connects to 1 community
- [[routing.py]] - degree 3, connects to 1 community