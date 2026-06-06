# Repository Index

This file provides links to all generated documentation files in the project context directory.

## Generated Documentation Files

1. [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) - High-level repository purpose, goals, maturity level, key innovations, and structure summary
2. [02_ARCHITECTURE.md](02_ARCHITECTURE.md) - Complete model architecture including layer stack, MLA, MoE, GDN, MTP details, embedding system, positional encoding, forward pass walkthrough, and parameter count estimates
3. [03_TRAINING_PIPELINE.md](03_TRAINING_PIPELINE.md) - End-to-end training flow, dataset preparation, data loading, curriculum learning, FSDP strategy, checkpointing, logging, optimizers, and schedulers
4. [04_DISTRIBUTED_SYSTEM.md](04_DISTRIBUTED_SYSTEM.md) - FSDP wrapping, sharding strategy, communication patterns, expert parallelism, expected cluster topology, scaling assumptions, and bottlenecks
5. [05_CONFIGURATION_REFERENCE.md](05_CONFIGURATION_REFERENCE.md) - Complete reference for all configuration options with default values, recommended values, and interactions between configs
6. [06_MEMORY_AND_PERFORMANCE.md](06_MEMORY_AND_PERFORMANCE.md) - VRAM analysis, activation memory, MLA savings, MoE memory behavior, checkpointing strategy, throughput optimizations, Triton kernels, and Flash attention usage
7. [07_RISKS_AND_TECHNICAL_DEBT.md](07_RISKS_AND_TECHNICAL_DEBT.md) - Bugs found, potential bugs, stability concerns, OOM risks, distributed training risks, architecture risks, and code quality concerns
8. [08_AGENT_PLAYBOOK.md](08_AGENT_PLAYBOOK.md) - Guidance for future agents on how to approach tasks, files to inspect first, critical components, common failure modes, safe refactoring guidelines, training debugging workflow, performance debugging workflow, and evaluation workflow
9. [09_CODEBASE_MAP.md](09_CODEBASE_MAP.md) - Directory-by-directory explanation, major files, dependencies between files, and call graph overview
10. [10_RESEARCH_ROADMAP.md](10_RESEARCH_ROADMAP.md) - Current architecture status, missing features, improvement opportunities, scaling roadmap, suggested experiments, and success metrics
11. [11_GLOSSARY.md](11_GLOSSARY.md) - Repository-specific terminology, acronyms, custom modules, and research references
12. [12_DECISION_LOG.md](12_DECISION_LOG.md) - Architectural decisions inferred from code, why each major component exists, and tradeoffs accepted

## How to Use This Index

Future agents can load any of these files as needed to understand specific aspects of the repository. For a quick start, begin with:

- [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) for high-level understanding
- [02_ARCHITECTURE.md](02_ARCHITECTURE.md) for model details
- [03_TRAINING_PIPELINE.md](03_TRAINING_PIPELINE.md) for training process
- [08_AGENT_PLAYBOOK.md](08_AGENT_PLAYBOOK.md) for guidance on making changes

All files are located in the `docs/project_context/` directory.