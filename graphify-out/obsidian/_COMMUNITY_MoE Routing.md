---
type: community
cohesion: 0.38
members: 10
---

# MoE Routing

**Cohesion:** 0.38 - loosely connected
**Members:** 10 nodes

## Members
- [[Benchmark MoE routing computation.          Args         dim Model dimension]] - rationale - benchmarks/benchmark_moe.py
- [[Benchmark routing overhead only (without expert computation).]] - rationale - benchmarks/benchmark_moe.py
- [[Compare MoE vs dense FFN computation.]] - rationale - benchmarks/benchmark_moe.py
- [[__init__.py]] - code - benchmarks/__init__.py
- [[benchmark_moe.py]] - code - benchmarks/benchmark_moe.py
- [[benchmark_moe_routing()]] - code - benchmarks/benchmark_moe.py
- [[benchmark_moe_vs_dense()]] - code - benchmarks/benchmark_moe.py
- [[benchmark_routing_overhead()]] - code - benchmarks/benchmark_moe.py
- [[dtype_1]] - code - benchmarks/benchmark_moe.py
- [[main()_2]] - code - benchmarks/benchmark_moe.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/MoE_Routing
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_DeepSeek MoE]]
- 3 edges to [[_COMMUNITY_Delta Rule]]
- 3 edges to [[_COMMUNITY_Training Benchmarks]]
- 2 edges to [[_COMMUNITY_Routing Gate]]

## Top bridge nodes
- [[__init__.py]] - degree 10, connects to 2 communities
- [[benchmark_moe.py]] - degree 7, connects to 2 communities
- [[benchmark_moe_routing()]] - degree 6, connects to 1 community
- [[benchmark_moe_vs_dense()]] - degree 6, connects to 1 community
- [[benchmark_routing_overhead()]] - degree 6, connects to 1 community