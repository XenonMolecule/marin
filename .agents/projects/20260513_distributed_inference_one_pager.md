# Distributed LLM Extraction Fleet — One Pager

*For the distributed-inference meeting — Michael Ryan, 2026-05-13.*

## What it is

A vLLM fleet that runs Qwen3-8B over **3,000 Common Crawl WARCs (~156M HTML
records)** on mixed-region preemptible TPUs. No queue, no scheduler, no
coordinator service: **workers coordinate exclusively through atomic GCS
writes**, and the whole system is re-entrant — kill it any time and it picks
up where it left off.

## Picture

```
       ┌────────────────────────┐
       │   Adaptive Launcher    │      Iris parent (CPU).
       │   launch_adaptive.py   │      probe → grow → backoff.
       └───────────┬────────────┘
                   │  N children, each --shuffle-seed=k
                   ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Worker A │   │ Worker B │…  │ Worker N │      One per TPU slice.
   │  vLLM    │   │  vLLM    │   │  vLLM    │      Different regions.
   │ region1  │   │ region2  │   │ regionN  │      All read same manifest.
   └────┬─────┘   └────┬─────┘   └────┬─────┘
        └──────────────┴──────────────┘
                       │  atomic writes (if_generation_match=0)
                       ▼
       gs://marin-{region}/{subdir}/data-{warc_hash}/
           _claimed                lease, 3h stale
           batch_NNNN.jsonl.gz     output
           batch_NNNN.count        record count sidecar
           batch_NNNN.tokens.gz    per-record FLOP stats
           _done                   terminal marker

       gs://marin-us-central1/{subdir}/_completed/data-{warc_hash}
                                       (cheap "what's done" listing)
```

## How it actually works (the four interesting bits)

1. **Atomic claim per WARC.** First writer of `_claimed` wins
   (`if_generation_match=0`). The file is refreshed between batches; a
   claim older than 3 h is reclaimable, again atomically.

2. **Per-batch checkpointing.** A WARC is processed in batches of 500
   records (~11 min on v5p-8). Each batch writes its output + sidecars
   immediately. Max work lost to preemption = one batch.

3. **Cross-region resume.** Workers in different regions write to their
   *local* regional bucket. Before/after every step, a worker lists **all
   six regional buckets** for the WARC. A batch existing anywhere means
   "done"; a `_done` anywhere means "skip". A single global `_completed/`
   registry of zero-byte markers is the fast path.

4. **Steal mode (endgame).** When most WARCs are claimed, workers stop
   trying to own WARCs and start claiming individual *batches* of
   in-progress WARCs from the back. Owners walk front-to-back, stealers
   walk back-to-front, they meet in the middle. Same atomic-write
   primitive, finer granularity.

## What's the meeting about

We've built this once. The mechanisms above are general enough to want as a
shared library, but the current code bakes them into the WARC/HTML domain.
Four decisions we should make:

- **Q1. GCS-as-coordinator, or a real coordinator service?** Today we pay
  6 cross-region list calls per WARC and a 3 h stale-claim heuristic
  because nobody runs a coordinator. Either we formalize the GCS primitive
  (atomic claim + lease + completion index) as first-class library API, or
  somebody owns Redis/Postgres/Iris-side state for the next 12 months.

- **Q2. What's the unit of work?** Right now the system has two:
  *shard* (WARC) for the happy path, *batch* for steal mode. A library
  built around nested shards from day one is more honest; a library
  around a single granularity is simpler. Pick one.

- **Q3. Where does the engine end and the application begin?** The
  current worker mixes vLLM, HTML truncation, chat templating, and
  `<think>`-token splitting in one function. A clean engine returns
  structured generations (`input_tokens, thinking_tokens, response_tokens,
  raw_text`); the application supplies a `(record)→messages` /
  `(raw)→record_or_None` IO contract. Do we commit to that split?

- **Q4. What's v0?** Concrete proposal to react to: ship **Engine** and
  **Coordinator** as a library; leave **Workload** and **IO contract** as
  user code. That captures the two pieces nobody wants to rewrite. The
  adaptive launcher is probably a separate utility (useful for any
  preemptible fleet, not just inference).

---

*Detailed spec, all 9 components, and 10 long-form discussion questions are
in `20260513_distributed_inference_explainer.md` if anyone wants the deep
dive.*
