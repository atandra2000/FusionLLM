---
source_file: "models/moe/moe.py"
type: "code"
community: "DeepSeek MoE"
location: "L20"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/DeepSeek_MoE
---

# DeepSeekMoE

## Connections
- [[.__init__()_9]] - `method` [EXTRACTED]
- [[._all_to_all_dispatch()]] - `method` [EXTRACTED]
- [[._compute_shared_experts()]] - `method` [EXTRACTED]
- [[._expert_forward_single()]] - `method` [EXTRACTED]
- [[._get_weighted_onehot()]] - `method` [EXTRACTED]
- [[._refresh_weight_stacks()]] - `method` [EXTRACTED]
- [[._try_grouped_gemm()]] - `method` [EXTRACTED]
- [[.forward()_4]] - `method` [EXTRACTED]
- [[.get_load_balance_loss()]] - `method` [EXTRACTED]
- [[.get_routing_stats()]] - `method` [EXTRACTED]
- [[.get_z_loss()]] - `method` [EXTRACTED]
- [[.update_gate_bias()]] - `method` [EXTRACTED]
- [[AuxLossFreeGate]] - `uses` [INFERRED]
- [[DeepSeekMoE]] - `uses` [INFERRED]
- [[DeepSeekMoE with shared experts and aux-loss-free load balancing.      Expert pa]] - `rationale_for` [EXTRACTED]
- [[Expert]] - `uses` [INFERRED]
- [[Tensor_1]] - `uses` [INFERRED]
- [[__init__.py_4]] - `imports` [EXTRACTED]
- [[__init__.py_5]] - `imports` [EXTRACTED]
- [[benchmark_moe.py]] - `imports` [EXTRACTED]
- [[benchmark_moe_routing()]] - `calls` [EXTRACTED]
- [[benchmark_moe_scaling()]] - `calls` [EXTRACTED]
- [[benchmark_moe_vectorized.py]] - `imports` [EXTRACTED]
- [[benchmark_moe_vs_dense()]] - `calls` [EXTRACTED]
- [[benchmark_moe_vs_dense()_1]] - `calls` [EXTRACTED]
- [[dtype_1]] - `uses` [INFERRED]
- [[moe.py]] - `contains` [EXTRACTED]
- [[transformer.py]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/DeepSeek_MoE