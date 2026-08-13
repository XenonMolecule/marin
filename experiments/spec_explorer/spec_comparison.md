# Spec comparison

| Spec | Stage | Note |
|---|---|---|
| **high_quality** | Single stage | One-prompt extraction with hard gates (English, ≥300 prose words) + positive criteria; deliberately strict, keeps ~1 in 10 pages. |
| **med_quality** | Single stage | One-prompt extraction with a looser-than-HQ quality bar that still filters nav/boilerplate, fragments, commerce, and incoherent pages. |
| **high_quality_v2** | Multi stage | Basically high_quality but two-stage (filter/quality-gate call, then a separate extraction call); better on fiction and QA/forum content. |
| **llm_pipeline_v1** | Multi stage | Two-stage: a filter/voting gate call, then a separate head + continuation extraction call. |
| **llm_simple_v1** | Single stage | The one-call variant — judging and extraction merged into a single call (plus a continuation call only for long docs). |
