---
type: community
cohesion: 0.08
members: 46
---

# Data Preparation

**Cohesion:** 0.08 - loosely connected
**Members:** 46 nodes

## Members
- [[Apply language + quality filter to an HF dataset, return ``Doc``.]] - rationale - data/prepare_data.py
- [[Collect ``(text, quality_score)`` pairs for one source.      Args         sourc]] - rationale - data/prepare_data.py
- [[Data package — see submodules for the actual surface.   ``data.dedup``        —]] - rationale - data/__init__.py
- [[Doc_1]] - code - data/prepare_data.py
- [[Heuristic quality score in 0, 2 (length + lexical diversity).]] - rationale - data/prepare_data.py
- [[Language-ID gate.          Uses fasttext ``lid.176`` model when available; falls]] - rationale - data/prepare_data.py
- [[Lazy import so this module can be imported without `transformers`.]] - rationale - data/prepare_data.py
- [[Load fasttext lid.176 model if available (gated by env-flag).]] - rationale - data/prepare_data.py
- [[Namespace]] - code - data/prepare_data.py
- [[PEP 562 lazy attribute access — keeps top-level imports cheap.]] - rationale - data/__init__.py
- [[Path_2]] - code - data/prepare_data.py
- [[Path_3]] - code - data/shard_writer.py
- [[Run the chosen dedup strategy.      Default is MinHash; ``strategy=prefix`` fo]] - rationale - data/prepare_data.py
- [[Tensor_3]] - code - data/prepare_data.py
- [[Tokenize documents and pack them into fixed-length sequences.      Pack strategy]] - rationale - data/prepare_data.py
- [[Write ``shardsmanifest.jsonl`` and return the path.]] - rationale - data/shard_writer.py
- [[Write a single shard atomically (temp + rename).]] - rationale - data/shard_writer.py
- [[Write token array to webdataset-style shards.      Each shard is a 256-byte head]] - rationale - data/shard_writer.py
- [[__getattr__()]] - code - data/__init__.py
- [[__init__.py_1]] - code - data/__init__.py
- [[_atomic_write_shard()]] - code - data/shard_writer.py
- [[_collect_cosmopedia()]] - code - data/prepare_data.py
- [[_collect_finemath()]] - code - data/prepare_data.py
- [[_collect_fineweb2()]] - code - data/prepare_data.py
- [[_collect_fineweb_edu()]] - code - data/prepare_data.py
- [[_collect_openr1_math()]] - code - data/prepare_data.py
- [[_collect_smollm_corpus()]] - code - data/prepare_data.py
- [[_collect_stack_edu()]] - code - data/prepare_data.py
- [[_filter_docs()]] - code - data/prepare_data.py
- [[_load_fasttext_lid()]] - code - data/prepare_data.py
- [[collect()]] - code - data/prepare_data.py
- [[deduplicate()]] - code - data/prepare_data.py
- [[export_eval_samples()]] - code - data/prepare_data.py
- [[load_tokenizer()]] - code - data/prepare_data.py
- [[main()_5]] - code - data/prepare_data.py
- [[normalize()]] - code - data/prepare_data.py
- [[np_dtype()]] - code - data/shard_writer.py
- [[parse_args()]] - code - data/prepare_data.py
- [[passes_language_filter()]] - code - data/prepare_data.py
- [[prepare_data.py]] - code - data/prepare_data.py
- [[prepare_data.py ===============  Dataset preparation pipeline for FusionLLM pre-]] - rationale - data/prepare_data.py
- [[quality_score()]] - code - data/prepare_data.py
- [[shard_writer.py]] - code - data/shard_writer.py
- [[tokenize_and_pack()]] - code - data/prepare_data.py
- [[write_manifest()]] - code - data/shard_writer.py
- [[write_shards()]] - code - data/shard_writer.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Data_Preparation
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Deduplication]]
- 1 edge to [[_COMMUNITY_Curriculum Shards]]

## Top bridge nodes
- [[prepare_data.py]] - degree 27, connects to 2 communities
- [[deduplicate()]] - degree 5, connects to 1 community