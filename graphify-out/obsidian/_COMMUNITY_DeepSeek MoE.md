---
type: community
cohesion: 0.14
members: 21
---

# DeepSeek MoE

**Cohesion:** 0.14 - loosely connected
**Members:** 21 nodes

## Members
- [[._all_to_all_dispatch()]] - code - models/moe/moe.py
- [[._compute_shared_experts()]] - code - models/moe/moe.py
- [[._expert_forward_single()]] - code - models/moe/moe.py
- [[._get_weighted_onehot()]] - code - models/moe/moe.py
- [[._try_grouped_gemm()]] - code - models/moe/moe.py
- [[.forward()_4]] - code - models/moe/moe.py
- [[.get_load_balance_loss()]] - code - models/moe/moe.py
- [[.get_routing_stats()]] - code - models/moe/moe.py
- [[.get_z_loss()]] - code - models/moe/moe.py
- [[.update_gate_bias()]] - code - models/moe/moe.py
- [[Args             x (T, dim) — flattened token representations         Returns]] - rationale - models/moe/moe.py
- [[Backward-compatible wrapper for dispatch.all_to_all_dispatch.]] - rationale - models/moe/moe.py
- [[Backward-compatible wrapper for dispatch.try_grouped_gemm.]] - rationale - models/moe/moe.py
- [[Build (Ttopk, E) one-hot assignment matrix weighted by routing scores.]] - rationale - models/moe/moe.py
- [[Compute all shared expert outputs and sum them.]] - rationale - models/moe/moe.py
- [[DeepSeekMoE_1]] - code - models/moe/moe.py
- [[DeepSeekMoE with shared experts and aux-loss-free load balancing.      Expert pa]] - rationale - models/moe/moe.py
- [[Router z-loss from the gate's cached pre-sigmoid logits.]] - rationale - models/moe/moe.py
- [[Single expert forward pass (SwiGLU or ReLU²).]] - rationale - models/moe/moe.py
- [[Tensor_14]] - code - models/moe/moe.py
- [[Update the gate's load-balancing bias using the cached token counts.]] - rationale - models/moe/moe.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/DeepSeek_MoE
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_MoE Benchmarks]]
- 5 edges to [[_COMMUNITY_Expert Forward]]
- 4 edges to [[_COMMUNITY_MoE Routing]]
- 2 edges to [[_COMMUNITY_Mamba Blocks]]
- 2 edges to [[_COMMUNITY_Expert Dispatch]]
- 2 edges to [[_COMMUNITY_Routing Gate]]

## Top bridge nodes
- [[DeepSeekMoE_1]] - degree 28, connects to 6 communities
- [[Tensor_14]] - degree 9, connects to 2 communities
- [[._expert_forward_single()]] - degree 5, connects to 1 community