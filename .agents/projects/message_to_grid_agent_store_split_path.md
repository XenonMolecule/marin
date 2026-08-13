# Heads-up: cell cache paths now carry the `train/` split level

**Date:** 2026-07-30
**From:** the OLMIX data-mixing work (`.claude/plans/…curried-moonbeam.md`)
**Touches:** `experiments/datakit/store/datakit_store.py`, `experiments/datakit/store/test_datakit_store.py`

## What was wrong

`_write_subshard_cache` wrote each cell to

```
<output>/cluster=<C>/quality=<Q>/sub=<S>
```

but Levanter resolves a mixture component by appending the split **itself**:

```python
# lib/levanter/src/levanter/data/text/datasets.py  (load_cache, and train_sets)
load_lm_dataset_cache(os.path.join(base_cache, split), self.format, tokenizer, ...)
```

So `DatasetComponent(cache_dir=".../quality=3/sub=0")` looks for
`.../quality=3/sub=0/train/shard_ledger.json`, finds nothing, and raises
*"No source and no cache found for component"*. Every normal marin cache has that
level — e.g. `gs://.../tokenized/Emilia/DE-c76e96/train/shard_ledger.json`.

This is exactly the class of failure your `verify_grid_store.py` docstring calls out
("a mixture will accept the path and train on nothing"), except it bites one level
higher up: the mixture won't even accept the path.

## What changed

* `_write_subshard_cache(..., split=)` now writes to `.../sub=<S>/<split>`.
* `_finalize_buckets(..., split=)` consolidates the k>1 case into
  `.../cluster=<C>/quality=<Q>/<split>`, so both branches produce the same shape.
* `BucketCacheStats.path` still means **the cache dir `TreeCache.load` takes** — so
  your `verify_grid_store.py` needs no change, and its tests still pass (verified:
  12 passed across `test_datakit_store.py` + `test_verify_grid_store.py`).
* Added `BucketCacheStats.component_cache_dir` — `path` minus the split level. That
  is what a Levanter `DatasetComponent.cache_dir` takes. Use it, don't hand-roll a
  `dirname`.
* Added `test_cell_path_carries_the_split_level_a_mixture_component_needs`,
  parametrized over `default_subshards=[1, 3]` to cover both the "cell *is* the sub
  cache" and consolidated branches. Confirmed it fails on the old path shape.

## Why it mattered to do it now

Nothing has been shuffled yet (`datakit/tokenize/` and `datakit/store/` are both
empty in `gs://marin-us-central1`), so this was a one-line change. After the shuffle
it would have meant re-running it or copying ~600 zarr trees. **If you had a
`grid_store` run in flight when you read this, check whether its cells have the
`train/` level and re-run if not.**

No other behaviour changed: token order, document boundaries, the positional join,
subshard planning, and the artifact schema are all untouched.
