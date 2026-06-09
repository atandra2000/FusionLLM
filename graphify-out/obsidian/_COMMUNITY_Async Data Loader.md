---
type: community
cohesion: 0.15
members: 24
---

# Async Data Loader

**Cohesion:** 0.15 - loosely connected
**Members:** 24 nodes

## Members
- [[.__del__()]] - code - data/async_loader.py
- [[.__enter__()]] - code - data/async_loader.py
- [[.__exit__()]] - code - data/async_loader.py
- [[.__iter__()_1]] - code - data/async_loader.py
- [[.__next__()]] - code - data/async_loader.py
- [[._iter_async()]] - code - data/async_loader.py
- [[._iter_sync()]] - code - data/async_loader.py
- [[._to_pair()]] - code - data/async_loader.py
- [[.set_batch_size()]] - code - data/async_loader.py
- [[.set_seq_len()]] - code - data/async_loader.py
- [[.set_shards()_1]] - code - data/async_loader.py
- [[.start()]] - code - data/async_loader.py
- [[.stats()]] - code - data/async_loader.py
- [[.stop()]] - code - data/async_loader.py
- [[Async iteration pull pre-paged micro-batches from the queue.]] - rationale - data/async_loader.py
- [[AsyncShardLoader]] - code - data/async_loader.py
- [[Replace the shard list and rebuild the index (curriculum hot-swap).]] - rationale - data/async_loader.py
- [[Reshape a flat int64 buffer to (batch, seqlen) and build targets.]] - rationale - data/async_loader.py
- [[Signal the async worker to exit and wait for it.]] - rationale - data/async_loader.py
- [[Start the async worker thread (no-op in sync mode).]] - rationale - data/async_loader.py
- [[Synchronous iteration read one micro-batch, yield, repeat.]] - rationale - data/async_loader.py
- [[Tensor_2]] - code - data/async_loader.py
- [[Two-stage async loader over the sharded mmap corpus.      Args         manifest]] - rationale - data/async_loader.py
- [[ndarray]] - code - data/async_loader.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Async_Data_Loader
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Async Sharding]]
- 2 edges to [[_COMMUNITY_Training Pipeline]]
- 2 edges to [[_COMMUNITY_Config Bundle]]
- 1 edge to [[_COMMUNITY_Shard Index]]
- 1 edge to [[_COMMUNITY_Scheduler & Setup]]

## Top bridge nodes
- [[AsyncShardLoader]] - degree 23, connects to 4 communities
- [[._iter_sync()]] - degree 6, connects to 1 community
- [[.set_shards()_1]] - degree 5, connects to 1 community