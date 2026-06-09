---
type: community
cohesion: 0.10
members: 26
---

# Delta Rule

**Cohesion:** 0.10 - loosely connected
**Members:** 26 nodes

## Members
- [[._delta_rule()]] - code - models/gated_deltanet.py
- [[.forward()]] - code - models/gated_deltanet.py
- [[Benchmark GatedDeltaNet delta-rule implementation.          Args         seqlen]] - rationale - benchmarks/benchmark_delta_rule.py
- [[Compare chunked vs sequential delta-rule implementations.          Note Sequent]] - rationale - benchmarks/benchmark_delta_rule.py
- [[Compute the delta-rule recurrence via chunked associative scan.      Args]] - rationale - kernels/delta_rule.py
- [[Process one chunk for one (batch, head) pair.          Each block handles one (b]] - rationale - kernels/delta_rule.py
- [[Pure PyTorch chunked delta-rule with parallel scan (no Triton required).      Th]] - rationale - kernels/delta_rule.py
- [[Reference delta-rule scan in pure PyTorch.          For each head h, the state h]] - rationale - models/gated_deltanet.py
- [[Return True if the triton kernel is available on this system.]] - rationale - kernels/delta_rule.py
- [[Return a list of autotune configs for the delta-rule chunk kernel.]] - rationale - kernels/delta_rule.py
- [[Tensor_6]] - code - kernels/delta_rule.py
- [[Tensor_9]] - code - models/gated_deltanet.py
- [[_autotune_configs()]] - code - kernels/delta_rule.py
- [[_delta_rule_chunk_kernel()]] - code - kernels/delta_rule.py
- [[_delta_rule_chunked_pytorch()]] - code - kernels/delta_rule.py
- [[benchmark_delta_rule()]] - code - benchmarks/benchmark_delta_rule.py
- [[benchmark_delta_rule.py]] - code - benchmarks/benchmark_delta_rule.py
- [[benchmark_delta_rule_vs_sequential()]] - code - benchmarks/benchmark_delta_rule.py
- [[chunked_delta_rule()]] - code - kernels/delta_rule.py
- [[constexpr_1]] - code - kernels/delta_rule.py
- [[delta_rule.py]] - code - kernels/delta_rule.py
- [[dtype]] - code - benchmarks/benchmark_delta_rule.py
- [[gated_deltanet.py]] - code - models/gated_deltanet.py
- [[has_triton()_1]] - code - kernels/delta_rule.py
- [[main()_1]] - code - benchmarks/benchmark_delta_rule.py
- [[x (bsz, seqlen, d_model)  →  (bsz, seqlen, d_model).]] - rationale - models/gated_deltanet.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Delta_Rule
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_MoE Routing]]
- 3 edges to [[_COMMUNITY_Transformer Blocks]]

## Top bridge nodes
- [[benchmark_delta_rule.py]] - degree 6, connects to 1 community
- [[benchmark_delta_rule()]] - degree 6, connects to 1 community
- [[benchmark_delta_rule_vs_sequential()]] - degree 5, connects to 1 community
- [[.forward()]] - degree 4, connects to 1 community
- [[gated_deltanet.py]] - degree 3, connects to 1 community