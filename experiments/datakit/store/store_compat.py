# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims so upstream's ``datakit_store`` runs in this checkout.

Everything here exists because this fork is ~800 commits behind upstream, not
because the store needed redesigning. Keeping the shims in their own module means
``datakit_store.py`` stays a recognisable copy of upstream's, so future diffs
remain readable.

Three groups:

* **Filesystem** — upstream uses ``rigging.filesystem.StoragePath``, which does
  not exist here. ``sp_open`` / ``sp_glob`` / ``sp_exists`` are fsspec equivalents.
* **Artifacts** — upstream's ``read_artifact`` / ``write_artifact`` round-trip a
  pydantic model to ``<dir>/.artifact.json``. Reimplemented directly on fsspec.
* **Input descriptors** — the store takes typed handles for its five inputs. The
  modules defining them are not ported, so these are the minimal shapes the store
  actually reads. Field names match upstream so the store body is untouched.

``deterministic_hash`` must be stable **across processes and runs**: it decides
which subshard a document lands in, so a per-process hash (Python's ``hash()``
with randomised seeding) would scatter one document's tokens across subshards on
a retry. blake2b gives a fixed 64-bit value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import fsspec

from experiments.fsspec_paths import fsspec_glob


def sp_open(path: str, mode: str = "rb"):
    """Open a local or remote path. Replaces ``StoragePath(path).open(mode)``."""
    return fsspec.open(path, mode).open()


def sp_glob(pattern: str) -> list[str]:
    """Sorted glob over a local or remote pattern. Replaces ``StoragePath(...).glob()``."""
    return sorted(fsspec_glob(pattern))


def sp_exists(path: str) -> bool:
    """Whether a local or remote path exists. Replaces ``StoragePath(path).exists()``."""
    fs, resolved = fsspec.core.url_to_fs(path)
    return bool(fs.exists(resolved))


def deterministic_hash(value: str) -> int:
    """Stable 64-bit hash, identical across processes and runs.

    Python's builtin ``hash()`` is seeded per process, so using it for subshard
    assignment would send the same document to different subshards on a retry —
    duplicating it in one and losing it from another.
    """
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def write_artifact(artifact, output_path: str) -> None:
    """Persist a pydantic model to ``<output_path>/.artifact.json``."""
    with fsspec.open(f"{output_path.rstrip('/')}/.artifact.json", "w") as fh:
        fh.write(artifact.model_dump_json(indent=2))


def read_artifact(output_path: str, model_cls):
    """Load a pydantic model previously written by :func:`write_artifact`."""
    with fsspec.open(f"{output_path.rstrip('/')}/.artifact.json") as fh:
        return model_cls(**json.load(fh))


# --- input descriptors -------------------------------------------------------
# Minimal stand-ins carrying only the fields datakit_store reads. Field names
# deliberately match upstream's so the store body needs no edits.


@dataclass(frozen=True)
class TokenizedAttrData:
    """Tokenized ``{id, input_ids}`` parquet, one file per attribute shard.

    ``source_main_dirs`` records which document tree each split was tokenized
    from; the store cross-checks it against the assignment table's own record so
    a mixed-provenance join fails loudly instead of silently mis-labelling docs.
    """

    output_dirs: dict[str, str]
    source_main_dirs: dict[str, str]
    tokenizer: str


@dataclass(frozen=True)
class DeconAttributes:
    """Decontamination attributes. Optional in this fork — see the store's fork note."""

    main_output_dir: str


@dataclass(frozen=True)
class AssignmentAttrData:
    """Domain assignment. For us the column is ``cluster_24`` from WebOrganizer.

    ``k_train`` / ``k_views`` exist so ``_validate_cluster_view`` can confirm the
    requested view was materialised; with a single 24-way view, ``k_train=24``.
    """

    output_dir: str
    source_main_dir: str
    k_train: int
    k_views: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class QualityScores:
    """Quality attributes with a precomputed, calibrated ``quality_bucket`` column.

    ``model_dir`` / ``calib_file`` are compared across sources so a store cannot
    mix buckets produced by two different scorers, which would make cell indices
    incomparable between corpora.
    """

    main_output_dir: str
    model_dir: str
    calib_file: str
    bucket_edges: list[float]


@dataclass(frozen=True)
class DedupSourceEntry:
    """Per-source fuzzy-dedup attribute directory."""

    attr_dir: str


@dataclass(frozen=True)
class FuzzyDupsAttrData:
    """Fuzzy-dedup attributes, keyed by source document dir. Optional in this fork.

    Our corpora were deduped upstream (high_quality) or deliberately not (the
    rest), so the store is normally passed ``dedup=None`` — see the store's fork
    note.
    """

    sources: dict[str, DedupSourceEntry] = field(default_factory=dict)
