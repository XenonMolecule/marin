# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OLMo "Base Easy" bits-per-byte (bpb) perplexity subset.

These are the gold-continuation bpb tasks from AI2's OlmoBaseEval suite
(allenai/OLMo-in-loop-evals), staged in GCS at
``eval_datasets/olmo_in_loop_evals/oe_eval_tasks/<task>/<variant>/`` per region.

Each entry is a ``"<task>/<variant>"`` directory name under ``oe_eval_tasks/``.
Every task here scores a single gold continuation per doc and reports
``bits_per_byte = -sum(logprob of continuation) / continuation_byte_len * log2(e)``
— a pure loglikelihood measurement, which is why these run through Levanter's
``loglikelihood`` path directly rather than the marin lm-eval-harness task registry
(the MC OLMES suite lives in ``olmes_base`` instead).

Membership is the bpb/perplexity core of Base Easy: code (humaneval/mbpp + the
multilingual mt_mbpp set), math (gsm8k + minerva_math subjects), and language/QA
(coqa, drop, jeopardy, lambada, naturalqs, squad). The exact OlmoBaseEval "Base
Easy" list should be pinned against the olmes suite config; until then this is the
defensible gold-bpb subset (all 116 variants are staged, so editing this tuple is
the only change needed to add/drop tasks).
"""

CODE_BPB = (
    "codex_humaneval/gold_bpb_0shot",
    "codex_humaneval/gold_bpb_3shot",
    "codex_mbpp/gold_bpb_0shot",
    "codex_mbpp/gold_bpb_3shot",
)

MT_MBPP_BPB = (
    "mt_mbpp_python/gold_bpb_3shot",
    "mt_mbpp_java/gold_bpb_3shot",
    "mt_mbpp_javascript/gold_bpb_3shot",
    "mt_mbpp_typescript/gold_bpb_3shot",
    "mt_mbpp_cpp/gold_bpb_3shot",
    "mt_mbpp_c/gold_bpb_3shot",
    "mt_mbpp_csharp/gold_bpb_3shot",
    "mt_mbpp_go/gold_bpb_3shot",
    "mt_mbpp_rust/gold_bpb_3shot",
    "mt_mbpp_ruby/gold_bpb_3shot",
    "mt_mbpp_php/gold_bpb_3shot",
    "mt_mbpp_r/gold_bpb_3shot",
    "mt_mbpp_bash/gold_bpb_3shot",
    "mt_mbpp_scala/gold_bpb_3shot",
    "mt_mbpp_swift/gold_bpb_3shot",
    "mt_mbpp_haskell/gold_bpb_3shot",
    "mt_mbpp_matlab/gold_bpb_3shot",
)

MATH_BPB = (
    "gsm8k/gold_bpb_5shot",
    "minerva_math_500/gold_bpb_0shot",
    "minerva_math_algebra/gold_bpb_0shot",
    "minerva_math_counting_and_probability/gold_bpb_0shot",
    "minerva_math_geometry/gold_bpb_0shot",
    "minerva_math_intermediate_algebra/gold_bpb_0shot",
    "minerva_math_number_theory/gold_bpb_0shot",
    "minerva_math_prealgebra/gold_bpb_0shot",
    "minerva_math_precalculus/gold_bpb_0shot",
)

QA_LANG_BPB = (
    "coqa/bpb_0shot",
    "drop/bpb_5shot",
    "jeopardy/bpb_5shot",
    "lambada/bpb_0shot",
    "naturalqs_open/bpb_5shot",
    "squad/bpb_5shot",
)

OLMO_BASE_EASY_BPB: tuple[str, ...] = CODE_BPB + MT_MBPP_BPB + MATH_BPB + QA_LANG_BPB

# The multiple-choice QA tasks of AI2's olmo3:base_easy qa_bpb, scored as rc:bpb (bits-per-byte
# of the gold answer choice) at the OLMES-standard shot counts (lab_bench is 3-shot). Together
# with the QA_LANG_BPB generation tasks these make up qa_bpb minus MMLU (absent from the in-loop
# bundle). coqa/drop/jeopardy/naturalqs/squad here use generation bpb, not AI2's gen2mc formulation.
QA_RC_MC_BPB: tuple[str, ...] = (
    "arc_challenge/rc_5shot",
    "arc_easy/rc_5shot",
    "csqa/rc_5shot",
    "hellaswag/rc_5shot",
    "winogrande/rc_5shot",
    "socialiqa/rc_5shot",
    "piqa/rc_5shot",
    "sciq/rc_5shot",
    "qasper_yesno/rc_5shot",
    "basic_skills_arithmetic/rc_5shot",
    "basic_skills_coding/rc_5shot",
    "basic_skills_common_knowledge/rc_5shot",
    "basic_skills_logical_reasoning/rc_5shot",
    "basic_skills_pattern/rc_5shot",
    "basic_skills_string_operations/rc_5shot",
    "lab_bench_dbqa/rc_3shot",
    "lab_bench_protocolqa/rc_3shot",
    "medmcqa/rc_5shot",
    "medqa_en/rc_5shot",
    "sciriff_yesno/rc_5shot",
)


# BLEnD everyday-cultural-knowledge, rc:bpb over the English MC set (gold = the target
# country's top-voted answer). Built by ``build_blend_bpb_requests.py``, one task per
# country. Diagnostics only — NOT part of the olmix mixture objective.
BLEND_BPB: tuple[str, ...] = (
    "blend_algeria/rc_5shot",
    "blend_assam/rc_5shot",
    "blend_azerbaijan/rc_5shot",
    "blend_china/rc_5shot",
    "blend_ethiopia/rc_5shot",
    "blend_greece/rc_5shot",
    "blend_indonesia/rc_5shot",
    "blend_iran/rc_5shot",
    "blend_mexico/rc_5shot",
    "blend_north_korea/rc_5shot",
    "blend_northern_nigeria/rc_5shot",
    "blend_south_korea/rc_5shot",
    "blend_spain/rc_5shot",
    "blend_uk/rc_5shot",
    "blend_us/rc_5shot",
    "blend_west_java/rc_5shot",
)


def resolve_tasks(spec: str) -> list[str]:
    """Resolve a --tasks spec into a list of ``"<task>/<variant>"`` dirs.

    Keywords: ``"all"`` → the 36 gold-continuation bpb tasks; ``"qa_rc"`` → the 20 MC-QA
    rc:bpb tasks; ``"all+qa_rc"`` → both; ``"blend"`` → the 16 BLEnD country tasks.
    Otherwise a comma-separated list of dir names.
    """
    if spec == "all":
        return list(OLMO_BASE_EASY_BPB)
    if spec == "qa_rc":
        return list(QA_RC_MC_BPB)
    if spec == "all+qa_rc":
        return list(OLMO_BASE_EASY_BPB) + list(QA_RC_MC_BPB)
    if spec == "blend":
        return list(BLEND_BPB)
    return [t.strip() for t in spec.split(",") if t.strip()]
