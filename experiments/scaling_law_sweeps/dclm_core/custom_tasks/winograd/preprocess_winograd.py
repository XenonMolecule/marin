# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Schema scoring for DCLM CORE's `winograd` (WSC273), mirroring lm-eval's native
# `winogrande` mechanism. DCLM/MosaicML rows: {context_options: [ctx0, ctx1],
# continuation, gold}. The "schema" task type scores P(continuation | ctx_i) for
# each option-substituted context and picks the argmax. In lm-eval this is done by
# INVERTING the usual multiple_choice roles: the choices are the varying contexts,
# the target is the shared continuation, and doc_to_text carries the gold index.


def doc_to_text(doc):
    return doc["gold"]


def doc_to_target(doc):
    return doc["continuation"].strip()


def doc_to_choice(doc):
    return doc["context_options"]
