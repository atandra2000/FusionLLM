---
type: community
cohesion: 0.16
members: 17
---

# Compile Benchmarks

**Cohesion:** 0.16 - loosely connected
**Members:** 17 nodes

## Members
- [[.compile_for_inference()]] - code - models/transformer.py
- [[.get_compiled_submodules()]] - code - models/transformer.py
- [[Benchmark forward pass.          Args         model Model to benchmark]] - rationale - benchmarks/benchmark_compile.py
- [[Benchmark torch.compile performance.          Args         dim Model dimension]] - rationale - benchmarks/benchmark_compile.py
- [[Compile the model for optimized inference.                  Args             mo]] - rationale - models/transformer.py
- [[Create test configuration for benchmarking.]] - rationale - benchmarks/benchmark_compile.py
- [[Get submodules suitable for compilation.                  Returns             D]] - rationale - models/transformer.py
- [[Module]] - code - benchmarks/benchmark_compile.py
- [[Run compile benchmarks.]] - rationale - benchmarks/benchmark_compile.py
- [[Tensor]] - code - benchmarks/benchmark_compile.py
- [[The full backbone. ``config`` is the ``model`` block of the YAML.]] - rationale - models/transformer.py
- [[Transformer]] - code - models/transformer.py
- [[benchmark_compile.py]] - code - benchmarks/benchmark_compile.py
- [[benchmark_compile_performance()]] - code - benchmarks/benchmark_compile.py
- [[benchmark_forward()]] - code - benchmarks/benchmark_compile.py
- [[create_test_config()]] - code - benchmarks/benchmark_compile.py
- [[main()]] - code - benchmarks/benchmark_compile.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Compile_Benchmarks
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_Mamba Blocks]]
- 4 edges to [[_COMMUNITY_Transformer Blocks]]
- 2 edges to [[_COMMUNITY_Parallel Embedding]]
- 2 edges to [[_COMMUNITY_Scheduler & Setup]]
- 2 edges to [[_COMMUNITY_Config Bundle]]
- 1 edge to [[_COMMUNITY_MLA Attention]]
- 1 edge to [[_COMMUNITY_MoLE Model]]
- 1 edge to [[_COMMUNITY_Training Pipeline]]

## Top bridge nodes
- [[Transformer]] - degree 23, connects to 8 communities
- [[benchmark_compile.py]] - degree 7, connects to 1 community
- [[.get_compiled_submodules()]] - degree 3, connects to 1 community