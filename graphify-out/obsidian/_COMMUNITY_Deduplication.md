---
type: community
cohesion: 0.11
members: 31
---

# Deduplication

**Cohesion:** 0.11 - loosely connected
**Members:** 31 nodes

## Members
- [[.__init__()_4]] - code - data/dedup.py
- [[.__init__()_3]] - code - data/dedup.py
- [[.__len__()_2]] - code - data/dedup.py
- [[._init_perms()]] - code - data/dedup.py
- [[.add()]] - code - data/dedup.py
- [[.candidates()]] - code - data/dedup.py
- [[.signature()]] - code - data/dedup.py
- [[Collapse all whitespace runs to a single space and strip.]] - rationale - data/dedup.py
- [[Compute a MinHash signature for a document.      The signature is a length-``num]] - rationale - data/dedup.py
- [[Compute the MinHash signature for one document.          Returns a length-``num_]] - rationale - data/dedup.py
- [[Deduplicate ``docs`` using the chosen ``strategy``.      Selection rule on dupli]] - rationale - data/dedup.py
- [[Deduplication primitives for the pre-training corpus.  Three strategies, in incr]] - rationale - data/dedup.py
- [[Doc]] - code - data/dedup.py
- [[Drop docs whose first ``n`` bytes match a previously seen doc.      Cheap (O(1)]] - rationale - data/dedup.py
- [[Exact Jaccard estimate from two MinHash signatures.]] - rationale - data/dedup.py
- [[LSH index over MinHash signatures.      Splits a signature into ``num_bands`` co]] - rationale - data/dedup.py
- [[LSHIndex]] - code - data/dedup.py
- [[MD5-of-first-128-words dedup.  Cheaper than MinHash, less precise.]] - rationale - data/dedup.py
- [[Match the pre-Phase-1 ``near_duplicate_key`` helper exactly.      Kept for the u]] - rationale - data/dedup.py
- [[MinHashHasher]] - code - data/dedup.py
- [[Tokenize on whitespace and yield ``n``-gram shingles as strings.]] - rationale - data/dedup.py
- [[Yield candidate duplicate doc-ids for ``doc_id``.          ``doc_id`` should no]] - rationale - data/dedup.py
- [[_jaccard()]] - code - data/dedup.py
- [[_normalize()]] - code - data/dedup.py
- [[_shingles()]] - code - data/dedup.py
- [[dedup.py]] - code - data/dedup.py
- [[deduplicate_docs()]] - code - data/dedup.py
- [[exact_prefix_dedup()]] - code - data/dedup.py
- [[md5_fallback()]] - code - data/dedup.py
- [[md5_fallback_key()]] - code - data/dedup.py
- [[ndarray_1]] - code - data/dedup.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Deduplication
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Data Preparation]]

## Top bridge nodes
- [[deduplicate_docs()]] - degree 13, connects to 1 community