---
type: community
cohesion: 0.29
members: 12
---

# MoE Benchmarks

**Cohesion:** 0.29 - loosely connected
**Members:** 12 nodes

## Members
- [[Benchmark MoE forward pass.          Args         moe MoE module         x In]] - rationale - benchmarks/benchmark_moe_vectorized.py
- [[Benchmark MoE scaling with different input sizes.          Args         dim Mo]] - rationale - benchmarks/benchmark_moe_vectorized.py
- [[Benchmark MoE vs dense FFN.          Args         dim Model dimension]] - rationale - benchmarks/benchmark_moe_vectorized.py
- [[Create MoE configuration for benchmarking.]] - rationale - benchmarks/benchmark_moe_vectorized.py
- [[DeepSeekMoE]] - code - benchmarks/benchmark_moe_vectorized.py
- [[Tensor_1]] - code - benchmarks/benchmark_moe_vectorized.py
- [[benchmark_moe_forward()]] - code - benchmarks/benchmark_moe_vectorized.py
- [[benchmark_moe_scaling()]] - code - benchmarks/benchmark_moe_vectorized.py
- [[benchmark_moe_vectorized.py]] - code - benchmarks/benchmark_moe_vectorized.py
- [[benchmark_moe_vs_dense()_1]] - code - benchmarks/benchmark_moe_vectorized.py
- [[create_moe_config()]] - code - benchmarks/benchmark_moe_vectorized.py
- [[main()_3]] - code - benchmarks/benchmark_moe_vectorized.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/MoE_Benchmarks
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_DeepSeek MoE]]

## Top bridge nodes
- [[benchmark_moe_scaling()]] - degree 7, connects to 1 community
- [[benchmark_moe_vs_dense()_1]] - degree 7, connects to 1 community
- [[benchmark_moe_vectorized.py]] - degree 6, connects to 1 community
- [[DeepSeekMoE]] - degree 4, connects to 1 community
- [[Tensor_1]] - degree 2, connects to 1 community