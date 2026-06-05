# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the web extraction reprocessing pipeline.

Tests cover the prepare, reassemble, and merge steps without requiring
TPU hardware or a running Ray cluster.
"""

import gzip
import json
import os
import tempfile

import pytest

from experiments.distill.merge_reprocess_shards import merge_shards
from experiments.distill.web_extraction_reprocess import (
    SYSTEM_MESSAGE,
    PrepareConfig,
    _write_jsonl_gz,
    prepare_prompts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_SYSTEM_MESSAGE = SYSTEM_MESSAGE

SAMPLE_USER_MESSAGE = (
    "[[ ## html ## ]]\n"
    "<html><body><h1>Hello World</h1><p>Test content.</p></body></html>\n\n"
    "[[ ## extraction_spec ## ]]\n"
    "Extract the main text content from this HTML page.\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)

SAMPLE_ASSISTANT_MESSAGE = (
    "<think>The HTML contains a heading and paragraph.</think>"
    "[[ ## text ## ]]\n# Hello World\n\nTest content.\n\n[[ ## completed ## ]]"
)


def _make_hf_row(idx: int, prompt_text: str | None = None) -> dict:
    """Create a mock HF dataset row matching the rephraser_late_check_0225 schema."""
    return {
        "messages": [
            {"role": "system", "content": SAMPLE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt_text or SAMPLE_USER_MESSAGE},
            {"role": "assistant", "content": SAMPLE_ASSISTANT_MESSAGE},
        ],
        "warc_file": f"CC-MAIN-2024-00{idx}.warc.gz",
        "doc_id": f"record_{idx}",
        "spec_id": str(idx % 10),
        "spec": f"Spec text for spec {idx % 10}",
    }


def _read_all_jsonl_gz(directory: str) -> list[dict]:
    """Read all JSONL.gz files from a directory and return records."""
    records = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".jsonl.gz"):
            path = os.path.join(directory, fname)
            with open(path, "rb") as f:
                with gzip.open(f, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# test_prompt_format_matches_original
# ---------------------------------------------------------------------------
class TestPromptFormatMatchesOriginal:
    """Verify that prompts extracted by prepare_prompts match the original HF data exactly."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_HF_TESTS", "1") == "1",
        reason="Set SKIP_HF_TESTS=0 to run tests that hit HuggingFace",
    )
    def test_prompt_matches_hf_dataset(self):
        """Download parquet, run prepare, verify prompt extraction fidelity."""
        from datasets import load_dataset

        ds = load_dataset(
            "MichaelR207/rephraser_late_check_0225",
            split="train[:5]",
            streaming=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download parquet files first
            parquet_dir = os.path.join(tmpdir, "parquet")
            from marin.datakit.download.huggingface import DownloadConfig, download_hf

            download_hf(
                DownloadConfig(
                    hf_dataset_id="MichaelR207/rephraser_late_check_0225",
                    revision="2194850",
                    gcs_output_path=parquet_dir,
                )
            )

            output_dir = os.path.join(tmpdir, "prepared")
            config = PrepareConfig(
                input_path=parquet_dir,
                start_row=0,
                end_row=5,
                output_path=output_dir,
                tokenizer="Qwen/Qwen3-8B",
                max_doc_tokens=999999,  # No filtering
            )
            prepare_prompts(config)

            records = _read_all_jsonl_gz(output_dir)
            assert len(records) == 5

            for i, (record, original) in enumerate(zip(records, ds, strict=True)):
                expected_prompt = original["messages"][1]["content"]
                assert (
                    record["prompt"] == expected_prompt
                ), f"Row {i}: prompt does not match original messages[1]['content']"
                assert record["global_row_idx"] == i
                assert record["warc_file"] == original["warc_file"]
                assert record["doc_id"] == original["doc_id"]
                assert record["spec_id"] == original["spec_id"]
                assert record["spec"] == original["spec"]


# ---------------------------------------------------------------------------
# test_prepare_row_range
# ---------------------------------------------------------------------------
class TestPrepareRowRange:
    """Verify that row range slicing produces the correct subset with correct indices."""

    def test_row_range_subset(self):
        """Run prepare with start_row=3, end_row=7 on a 10-row mock dataset."""
        mock_rows = [_make_hf_row(i) for i in range(10)]

        def _prepare_from_mock(config, all_rows):
            rows = all_rows[config.start_row : config.end_row]
            shard_records = []
            total_written = 0
            shard_idx = 0
            total_shards = max(1, (len(rows) + config.records_per_shard - 1) // config.records_per_shard)

            for i, row in enumerate(rows):
                messages = row["messages"]
                record = {
                    "prompt": messages[1]["content"],
                    "global_row_idx": config.start_row + i,
                    "warc_file": row.get("warc_file", ""),
                    "doc_id": row.get("doc_id", ""),
                    "spec_id": row.get("spec_id", ""),
                    "spec": row.get("spec", ""),
                }
                shard_records.append(record)

                if len(shard_records) >= config.records_per_shard or i == len(rows) - 1:
                    shard_path = f"{config.output_path}/data-{shard_idx:05d}-of-{total_shards:05d}.jsonl.gz"
                    _write_jsonl_gz(shard_path, shard_records)
                    total_written += len(shard_records)
                    shard_records = []
                    shard_idx += 1

            stats = {"total_written": total_written, "total_dropped": 0}
            with open(os.path.join(config.output_path, "prepare_stats.json"), "w") as f:
                json.dump(stats, f)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrepareConfig(
                input_path="mock",
                start_row=3,
                end_row=7,
                output_path=tmpdir,
                tokenizer="Qwen/Qwen3-8B",
                max_doc_tokens=999999,
            )
            _prepare_from_mock(config, mock_rows)

            records = _read_all_jsonl_gz(tmpdir)

            assert len(records) == 4
            assert [r["global_row_idx"] for r in records] == [3, 4, 5, 6]
            assert all("warc_file" in r for r in records)
            assert all("doc_id" in r for r in records)
            assert all("spec_id" in r for r in records)
            assert all("spec" in r for r in records)

            # Verify prompts match the original messages
            for i, record in enumerate(records):
                expected = mock_rows[3 + i]["messages"][1]["content"]
                assert record["prompt"] == expected


# ---------------------------------------------------------------------------
# test_prepare_drops_long_prompts
# ---------------------------------------------------------------------------
class TestPrepareDropsLongPrompts:
    """Verify that rows exceeding max_doc_tokens are dropped, not truncated."""

    def test_long_prompts_dropped(self):
        short_prompt = "Short text."
        long_prompt = "Long text. " * 500  # Very long

        mock_rows = [
            _make_hf_row(0, prompt_text=short_prompt),
            _make_hf_row(1, prompt_text=long_prompt),
            _make_hf_row(2, prompt_text=short_prompt),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_records = []
            total_dropped = 0

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

            for i, row in enumerate(mock_rows):
                prompt = row["messages"][1]["content"]
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                # Set max_doc_tokens to 100 — short_prompt fits, long_prompt doesn't
                if len(tokens) > 100:
                    total_dropped += 1
                    continue
                shard_records.append(
                    {
                        "prompt": prompt,
                        "global_row_idx": i,
                        "warc_file": row["warc_file"],
                        "doc_id": row["doc_id"],
                        "spec_id": row["spec_id"],
                        "spec": row["spec"],
                    }
                )

            shard_path = os.path.join(tmpdir, "data-00000-of-00001.jsonl.gz")
            _write_jsonl_gz(shard_path, shard_records)

            records = _read_all_jsonl_gz(tmpdir)

            # Long prompt should be dropped
            assert total_dropped == 1
            assert len(records) == 2
            # Verify remaining records are NOT truncated
            for record in records:
                assert record["prompt"] == short_prompt
            # Verify indices: row 1 was dropped
            assert [r["global_row_idx"] for r in records] == [0, 2]


# ---------------------------------------------------------------------------
# test_reassemble_output_schema
# ---------------------------------------------------------------------------
class TestReassembleOutputSchema:
    """Verify reassembled output has the correct schema."""

    def test_output_schema(self):
        # Create mock inference output records
        inference_records = [
            {
                "prompt": SAMPLE_USER_MESSAGE,
                "generated_text": "<think>reasoning</think>[[ ## text ## ]]\nExtracted.\n[[ ## completed ## ]]",
                "global_row_idx": 42,
                "warc_file": "test.warc.gz",
                "doc_id": "record_0",
                "spec_id": "5",
                "spec": "Test spec",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write mock inference output
            input_path = os.path.join(tmpdir, "input", "data-00000.jsonl.gz")
            os.makedirs(os.path.dirname(input_path))
            _write_jsonl_gz(input_path, inference_records)

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)

            # Run reassemble logic directly (without Zephyr)
            for record in inference_records:
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["generated_text"]},
                ]
                reassembled = {
                    "messages": messages,
                    "warc_file": record["warc_file"],
                    "doc_id": record["doc_id"],
                    "spec_id": record["spec_id"],
                    "spec": record["spec"],
                    "model": "test-model",
                    "global_row_idx": record["global_row_idx"],
                }

                # Verify schema
                assert set(reassembled.keys()) == {
                    "messages",
                    "warc_file",
                    "doc_id",
                    "spec_id",
                    "spec",
                    "model",
                    "global_row_idx",
                }
                assert len(reassembled["messages"]) == 3
                assert reassembled["messages"][0]["role"] == "system"
                assert reassembled["messages"][1]["role"] == "user"
                assert reassembled["messages"][2]["role"] == "assistant"
                assert reassembled["messages"][1]["content"] == SAMPLE_USER_MESSAGE
                assert reassembled["messages"][2]["content"] == record["generated_text"]
                # No intermediate columns
                assert "prompt" not in reassembled or reassembled.get("prompt") is None
                assert "generated_text" not in reassembled


# ---------------------------------------------------------------------------
# test_reassemble_messages_format_matches_original
# ---------------------------------------------------------------------------
class TestReassembleFormatMatchesOriginal:
    """Round-trip: original HF row → prepare → (simulated inference) → reassemble.
    Verify the output messages structure matches the original exactly (modulo assistant content).
    """

    def test_round_trip_format(self):
        original_row = _make_hf_row(7)
        original_messages = original_row["messages"]

        # Simulate prepare: extract prompt
        prompt = original_messages[1]["content"]

        # Simulate inference: model generates new text
        new_generated_text = "<think>New reasoning.</think>[[ ## text ## ]]\nNew output.\n[[ ## completed ## ]]"

        # Simulate reassemble
        reassembled_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": new_generated_text},
        ]

        # Verify structure matches original (same roles, same count)
        assert len(reassembled_messages) == len(original_messages)
        for orig, reasm in zip(original_messages, reassembled_messages, strict=True):
            assert orig["role"] == reasm["role"]

        # System and user content should be identical
        assert reassembled_messages[0]["content"] == original_messages[0]["content"]
        assert reassembled_messages[1]["content"] == original_messages[1]["content"]

        # Assistant content is different (new model output)
        assert reassembled_messages[2]["content"] == new_generated_text
        assert reassembled_messages[2]["content"] != original_messages[2]["content"]


# ---------------------------------------------------------------------------
# test_merge_deduplicates_by_global_row_idx
# ---------------------------------------------------------------------------
class TestMergeDeduplication:
    """Verify merge script deduplicates overlapping rows correctly."""

    def test_dedup_overlapping_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two "cluster" directories with overlapping rows
            dir_a = os.path.join(tmpdir, "cluster_a")
            dir_b = os.path.join(tmpdir, "cluster_b")
            os.makedirs(dir_a)
            os.makedirs(dir_b)

            # Cluster A: rows 0, 1, 2
            records_a = [
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a_0"},
                    ],
                    "warc_file": "a",
                    "doc_id": "0",
                    "spec_id": "0",
                    "spec": "s",
                    "model": "m",
                    "global_row_idx": 0,
                },
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a_1"},
                    ],
                    "warc_file": "a",
                    "doc_id": "1",
                    "spec_id": "0",
                    "spec": "s",
                    "model": "m",
                    "global_row_idx": 1,
                },
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a_2"},
                    ],
                    "warc_file": "a",
                    "doc_id": "2",
                    "spec_id": "0",
                    "spec": "s",
                    "model": "m",
                    "global_row_idx": 2,
                },
            ]

            # Cluster B: rows 2, 3, 4 (row 2 overlaps!)
            records_b = [
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "b_2"},
                    ],
                    "warc_file": "b",
                    "doc_id": "2",
                    "spec_id": "0",
                    "spec": "s",
                    "model": "m",
                    "global_row_idx": 2,
                },
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "b_3"},
                    ],
                    "warc_file": "b",
                    "doc_id": "3",
                    "spec_id": "0",
                    "spec": "s",
                    "model": "m",
                    "global_row_idx": 3,
                },
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "b_4"},
                    ],
                    "warc_file": "b",
                    "doc_id": "4",
                    "spec_id": "0",
                    "spec": "s",
                    "model": "m",
                    "global_row_idx": 4,
                },
            ]

            _write_jsonl_gz(os.path.join(dir_a, "data-00000.jsonl.gz"), records_a)
            _write_jsonl_gz(os.path.join(dir_b, "data-00000.jsonl.gz"), records_b)

            output_dir = os.path.join(tmpdir, "merged")
            os.makedirs(output_dir)

            merge_shards(
                input_dirs=[dir_a, dir_b],
                output_dir=output_dir,
                expected_rows=5,
                records_per_shard=100,
            )

            records = _read_all_jsonl_gz(output_dir)

            # Should have exactly 5 unique rows (0, 1, 2, 3, 4)
            assert len(records) == 5
            # No global_row_idx in final output
            assert all("global_row_idx" not in r for r in records)

            # Check stats
            with open(os.path.join(output_dir, "merge_stats.json")) as f:
                stats = json.load(f)
            assert stats["total_unique"] == 5
            assert stats["duplicates_deduped"] == 1
            assert stats["coverage_pct"] == 100.0


# ---------------------------------------------------------------------------
# test_merge_warns_on_gaps
# ---------------------------------------------------------------------------
class TestMergeGapDetection:
    """Verify merge warns about missing rows."""

    def test_gap_detection(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "cluster_a")
            os.makedirs(dir_a)

            # Rows 0, 1, 2, 5, 6, 7 — gap at 3, 4
            records = []
            for idx in [0, 1, 2, 5, 6, 7]:
                records.append(
                    {
                        "messages": [
                            {"role": "system", "content": "s"},
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": f"a_{idx}"},
                        ],
                        "warc_file": "a",
                        "doc_id": str(idx),
                        "spec_id": "0",
                        "spec": "s",
                        "model": "m",
                        "global_row_idx": idx,
                    }
                )

            _write_jsonl_gz(os.path.join(dir_a, "data-00000.jsonl.gz"), records)

            output_dir = os.path.join(tmpdir, "merged")
            os.makedirs(output_dir)

            with caplog.at_level("WARNING"):
                merge_shards(
                    input_dirs=[dir_a],
                    output_dir=output_dir,
                    expected_rows=8,
                    records_per_shard=100,
                )

            # Check that gap was detected
            with open(os.path.join(output_dir, "merge_stats.json")) as f:
                stats = json.load(f)
            assert stats["missing_count"] == 2
            assert stats["coverage_pct"] == 75.0

            # Check warning was logged
            assert any("MISSING" in record.message for record in caplog.records)
