---
source_file: "models/transformer.py"
type: "code"
community: "Compile Benchmarks"
location: "L252"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Compile_Benchmarks
---

# Transformer

## Connections
- [[.__init__()_19]] - `method` [EXTRACTED]
- [[.__init__()_33]] - `calls` [EXTRACTED]
- [[._get_checkpoint_policy()]] - `method` [EXTRACTED]
- [[.compile_for_inference()]] - `method` [EXTRACTED]
- [[.forward()_14]] - `method` [EXTRACTED]
- [[.forward_with_hidden()]] - `method` [EXTRACTED]
- [[.get_compiled_submodules()]] - `method` [EXTRACTED]
- [[.moe_layers()_1]] - `method` [EXTRACTED]
- [[ConfigBundle_6]] - `uses` [INFERRED]
- [[GatedDeltaNet]] - `uses` [INFERRED]
- [[Mamba2Block]] - `uses` [INFERRED]
- [[MoLE]] - `uses` [INFERRED]
- [[Module]] - `uses` [INFERRED]
- [[MultiHeadLatentAttention]] - `uses` [INFERRED]
- [[Pretrainer]] - `uses` [INFERRED]
- [[Tensor]] - `uses` [INFERRED]
- [[Tensor_25]] - `uses` [INFERRED]
- [[The full backbone. ``config`` is the ``model`` block of the YAML.]] - `rationale_for` [EXTRACTED]
- [[__init__.py_4]] - `imports` [EXTRACTED]
- [[benchmark_compile.py]] - `imports` [EXTRACTED]
- [[benchmark_compile_performance()]] - `calls` [EXTRACTED]
- [[trainer.py]] - `imports` [EXTRACTED]
- [[transformer.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Compile_Benchmarks