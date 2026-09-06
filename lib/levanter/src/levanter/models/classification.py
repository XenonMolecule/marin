# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Architecture-agnostic sequence-classification machinery.

``levanter.main.train_classifier`` trains any registered architecture: a config type maps to a
builder (and a saver) via :func:`register_classifier_arch`, and the trainer dispatches on the
config's type (walking the MRO, so config subclasses inherit their base's registration unless they
register their own).

Model contract (duck-typed, see ``ModernBertForSequenceClassification`` for the reference):

- ``model(tokens, attn_mask, *, key=None) -> NamedArray`` with an axis literally named ``"label"``
  (the eval path softmaxes over ``axis="label"``).
- ``model.compute_loss(example, *, key=None, reduction=hax.mean, reduction_axis=None)``.

Config contract: ``max_Pos`` (from ``LmConfig``) and ``pad_token_id`` (optional; the trainer falls
back to the tokenizer's pad id).
"""

import dataclasses
import enum
import json
import os
import tempfile
import typing
from typing import Callable, Optional

import equinox as eqx
import fsspec
from haliax import NamedArray

from levanter.layers.attention import AttentionMask


class ClassificationExample(eqx.Module):
    """A single (tokens, label) example for sequence classification.

    ``attn_mask`` is bidirectional by default (encoder); pad positions are handled via
    ``segment_ids`` on the mask rather than truncation so XLA shapes stay static.
    """

    tokens: NamedArray
    label: NamedArray  # scalar int (no Pos axis)
    attn_mask: AttentionMask | NamedArray | None = None

    @staticmethod
    def init(
        tokens: NamedArray,
        label: NamedArray,
        *,
        attn_mask: AttentionMask | NamedArray | None = None,
    ) -> "ClassificationExample":
        return ClassificationExample(tokens=tokens, label=label, attn_mask=attn_mask)


# builder: (config, Vocab, *, key, warm_start, axis_mapping, compute_dtype) -> classifier model
# saver: (config, model, path) -> None  (must run inside the Trainer mesh, like the models are)
_BUILDERS: dict[type, Callable] = {}
_SAVERS: dict[type, Callable] = {}


def register_classifier_arch(config_cls: type, *, build: Callable, save: Optional[Callable] = None) -> None:
    """Register a classifier architecture for ``config_cls`` (and, via MRO dispatch, its subclasses)."""
    _BUILDERS[config_cls] = build
    if save is not None:
        _SAVERS[config_cls] = save


def _lookup(registry: dict[type, Callable], config, what: str) -> Callable:
    for klass in type(config).__mro__:
        if klass in registry:
            return registry[klass]
    raise ValueError(
        f"no classifier {what} registered for config type {type(config).__name__}; "
        f"registered: {[c.__name__ for c in registry]}"
    )


def build_classifier(config, Vocab, *, key, warm_start: bool, axis_mapping=None, compute_dtype=None):
    """Build (and optionally warm-start) the classifier for ``config`` via the registered builder."""
    build = _lookup(_BUILDERS, config, "builder")
    return build(config, Vocab, key=key, warm_start=warm_start, axis_mapping=axis_mapping, compute_dtype=compute_dtype)


def save_classifier(config, model, path: str) -> None:
    """Save the trained classifier via the registered saver (HF export for HF-compatible archs)."""
    save = _lookup(_SAVERS, config, "saver")
    save(config, model, path)


def save_eqx_classifier(config, model, path: str) -> None:
    """Generic saver for non-HF architectures: equinox leaves + the config as JSON.

    Writes ``{path}/model.eqx`` and ``{path}/config.json``. eqx needs a local file, so serialize to
    a temp file and copy bytes (works for gs:// paths). Load with :func:`load_eqx_classifier`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "model.eqx")
        eqx.tree_serialise_leaves(local, model)
        with open(local, "rb") as src, fsspec.open(f"{path}/model.eqx", "wb") as dst:
            dst.write(src.read())
    with fsspec.open(f"{path}/config.json", "w") as f:
        json.dump({"config_class": type(config).__name__, **dataclasses.asdict(config)}, f, default=str)


def _enum_type(hint) -> Optional[type]:
    """The Enum class named by ``hint``, unwrapping Optional/Union; ``None`` if there isn't one."""
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint
    for arg in typing.get_args(hint):
        if isinstance(arg, type) and issubclass(arg, enum.Enum):
            return arg
    return None


def load_eqx_config(path: str):
    """Reconstruct the config written by :func:`save_eqx_classifier`, with enums restored.

    ``json.dump`` writes a ``StrEnum`` member as a bare string, so a naive ``ConfigCls(**raw)`` hands
    back ``str`` where the model expects the enum and ``FunnelBertConfig(**raw)`` fails on
    ``activation_function``. Coerce by the field's declared type so every consumer gets a real config
    instead of hand-rolling the same coercion.
    """
    with fsspec.open(f"{path}/config.json", "r") as f:
        raw = json.load(f)
    name = raw.pop("config_class", None)
    by_name = {cls.__name__: cls for cls in _BUILDERS}
    if name not in by_name:
        raise ValueError(f"config_class {name!r} is not a registered arch; registered: {sorted(by_name)}")
    config_cls = by_name[name]
    hints = typing.get_type_hints(config_cls)
    kwargs = {}
    for field in dataclasses.fields(config_cls):
        if not field.init or field.name not in raw:
            continue
        value = raw[field.name]
        enum_cls = _enum_type(hints.get(field.name))
        kwargs[field.name] = enum_cls(value) if enum_cls is not None and isinstance(value, str) else value
    return config_cls(**kwargs)


def load_eqx_classifier(template, path: str):
    """Deserialize ``{path}/model.eqx`` into ``template`` (a model pytree with matching structure)."""
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "model.eqx")
        with fsspec.open(f"{path}/model.eqx", "rb") as src, open(local, "wb") as dst:
            dst.write(src.read())
        return eqx.tree_deserialise_leaves(local, template)
