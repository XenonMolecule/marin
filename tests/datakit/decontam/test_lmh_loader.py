# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from experiments.datakit.decontam import lmh_loader


def test_legacy_dataset_names_resolve_to_their_hub_repo(monkeypatch, tmp_path):
    calls = []

    def dataset_info(self, name):
        calls.append(name)
        return SimpleNamespace(id={"gsm8k": "openai/gsm8k"}[name])

    monkeypatch.setattr(lmh_loader.HfApi, "dataset_info", dataset_info)
    lmh_loader.canonical_dataset_id.cache_clear()

    assert lmh_loader.canonical_dataset_id("gsm8k") == "openai/gsm8k"
    assert lmh_loader.canonical_dataset_id("allenai/winogrande") == "allenai/winogrande"
    assert lmh_loader.canonical_dataset_id("json") == "json"
    assert lmh_loader.canonical_dataset_id(str(tmp_path)) == str(tmp_path)
    assert calls == ["gsm8k"], "only no-namespace Hub names hit the API"
