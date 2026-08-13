# Quality × Domain grid — topic axis COMPLETE (2026-07-29)

Overnight run of the M (domain) axis over the four priority 10k corpora. **Zero data
loss, zero unresolved failures.** The N (quality) axis is still blocked on the
`pooled_junkgate2` artifact; nothing here depends on it, and it attaches later by `id` join.

## What landed

| corpus | shards | documents | gte tokens | layer |
|---|---:|---:|---:|---|
| high_quality_10k | 512 | 19,968,996 | 22,449,230,081 | **post-dedup + post-decon** |
| dclm_10k | 100 | 5,931,419 | 6,875,799,530 | pre-decon |
| nemotron_full_10k | 24,390 | 17,758,166 | 10,214,083,115 | pre-decon |
| fineweb_edu_10k | 1,469 | 2,349,555 | 2,284,607,100 | pre-decon |
| **total** | **26,471** | **46,008,136** | **41,823,719,826** | |

Artifacts:

```
gs://marin-us-central1/datakit/cluster_assign/{corpus}_gridv1/*.parquet
    id, native_id, url, cluster_24, topic_prob, token_length     # one file per input shard
gs://marin-us-central1/metadata/grid_v1/{corpus}/distribution.json
    counts + doc shares + token shares, per topic
gs://marin-us-central1/mirror/grid_v1/{corpus}/                   # dclm, fineweb_edu, nemotron
```

## Correctness evidence

- **high_quality returned exactly 19,968,996 documents**, matching the HF export's known
  corpus size to the document. Every document labelled once, no drops, no duplicates.
- Every corpus: `sum(doc shares) == 1.0000`, and done-markers == parquet shards == counts
  files == source shard count.
- fineweb_edu is 1469/1469/1469 **including its 7 empty shards**, which emit zero-row
  files rather than being skipped — the co-partitioning contract holds across the corpus.
- Topic distributions reproduce the earlier independent 1M-doc peek for high_quality
  (Health 8.35 vs 8.45, Finance 8.21 vs 8.47, Politics 7.79 vs 8.08, Adult 0.113 vs 0.07).

## Result: the four corpora have sharply different topic profiles

Token share (%), sorted by high_quality. This is the training-relevant unit; doc share
differs and is also stored.

| topic | high_quality | dclm | nemotron | fineweb_edu |
|---|---:|---:|---:|---:|
| Politics | 8.56 | **12.01** | 5.98 | 5.75 |
| Health | 8.46 | 8.37 | 9.08 | **19.06** |
| Finance & Business | 8.07 | 6.58 | 7.32 | 2.47 |
| Education & Jobs | 7.44 | 3.70 | 7.41 | 8.88 |
| Software Dev. | **7.43** | 4.58 | 2.36 | 1.87 |
| Science & Tech. | 6.96 | 8.04 | 6.45 | **21.99** |
| Entertainment | 5.90 | 8.80 | 6.39 | 0.93 |
| History | 3.11 | 2.90 | 2.48 | **11.40** |
| Literature | 2.77 | **6.67** | 4.09 | 4.22 |
| Social Life | 1.68 | 4.41 | 4.55 | 0.80 |
| Travel | 1.48 | 1.21 | **3.97** | 0.79 |
| Fashion & Beauty | 0.82 | 0.82 | **2.14** | 0.22 |
| **Adult** | 0.13 | 1.14 | **1.74** | **0.01** |

Four things worth carrying into the mixing design:

1. **Adult spans 174×** across corpora: fineweb_edu 0.01% → high_quality 0.13% → dclm
   1.14% → nemotron 1.74%. The LLM extraction filter cuts Adult ~9× below dclm without
   being told anything about topics; fineweb_edu's educational classifier cuts it ~100×
   further still.
2. **fineweb_edu is extraordinarily narrow.** Science & Tech + Health + History = **52%**
   of its tokens. Entertainment 0.93%, Games 0.33%, Fashion 0.22%. It is not a
   general-purpose corpus and will distort any mixture it enters at high weight.
3. **high_quality is the most balanced and the most code-rich.** Software Dev. 7.43% —
   1.6× dclm, 3.1× nemotron, 4× fineweb_edu — while keeping the flattest overall profile.
   If you want code and technical prose without fineweb_edu's narrowness, it is the source.
4. **nemotron leans consumer/lifestyle**: Home & Hobbies 6.27%, Travel 3.97%, Fashion
   2.14%, all the highest of the four.

Doc share and token share diverge enough to matter: dclm Politics is 9.29% of documents
but **12.01% of tokens**, Literature 4.99% → 6.67%. Mixing on document counts
systematically under-weights long-form topics.

## Operational findings (recorded in code)

- **Only `v5p-8` schedules as one task.** Iris derives replicas as chips/8, so v5p-16/32
  are gang-scheduled. Single-host requests were ASSIGNED in <30 s every time; 32 gang
  tasks sat PENDING 12 min and never placed, despite 116 preemptible hosts in the group.
  Gang capacity is not reachable by preempting scattered batch workers.
- **`interactive` priority is the whole game.** The extraction fleet runs its TPU children
  at `batch`, so anything above `batch` preempts them automatically and their coordinators
  re-absorb on release. No negotiation was needed. At `batch` we would have queued forever.
- **Merges must run in-region.** gcsfs cannot complete a TLS handshake to
  storage.googleapis.com from the dev laptop, though `gcloud` can.
- Egress was **$0.76 one-time** for 37.8 GB of mirrors (dclm + fineweb_edu + nemotron);
  high_quality is native to us-central1. The mirrors are permanent, so re-runs are free.

## Bug found and fixed

`grid_label` treated a zero-record shard as fatal. 7 of fineweb_edu's 1,469 shards are
20-byte empty gzips — WARCs from which nothing survived the upstream filter, which is
normal for a filtered corpus.

Failing was wrong, but *skipping* would have been worse: emitting 1,462 files for 1,469
inputs silently mis-aligns `datakit_store`'s positional join for every later shard. The fix
emits a zero-row file with an identical schema. Covered by two tests.

Also fixed: merge read shard tallies serially, so nemotron's 24,390 tallies took ~30 min of
pure round-trip latency. Now 32-way parallel — which matters far more for the eventual
quality re-merge, where the cross-tab must read 24,390 topic **and** 24,390 quality parquets.

## Next

1. **Quality axis** — blocked only on `pooled_junkgate2` (message drafted at
   `message_to_rav_quality_model.md`). Then:
   `launch_grid label --dataset X --stages quality --quality-model gs://...` — CPU-only,
   no TPU, and it re-merges into the full 24 × 5 grid with no topic work redone.
2. **fineweb_cc** — gated on the in-flight rebuild finishing.
3. **resiliparse** — needs the 10,364-WARC post-decon tree rebuilt first (it does not exist
   in any region), plus sign-off on $12.25 egress.
4. **Not yet done:** id-join coverage measurement on high_quality (plan Phase 0 step 7).
