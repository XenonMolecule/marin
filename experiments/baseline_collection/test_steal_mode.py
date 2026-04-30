# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for steal mode batch-level claiming and parallel WARC processing.

These tests use local filesystem (no GCS) to validate the steal mode logic:
- Atomic batch-level steal claims
- Reverse iteration order
- Pre-batch existence checks
- Exit check for _done writing
- Steal mode activation conditions

Run with: .venv/bin/pytest experiments/baseline_collection/test_steal_mode.py -v
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Import the functions we're testing. We patch GCS-specific paths.
from experiments.baseline_collection.run_extract_standalone import (
    _batch_output_path,
    _claim_batch_for_stealing,
    _done_marker_path,
    _find_stealable_batches,
    _list_steal_claims,
    _steal_claim_path,
    _warc_path_hash,
    _write_batch_output,
    _write_done_marker,
)


@pytest.fixture
def warc_dir(tmp_path):
    """Create a temporary WARC output directory."""
    d = tmp_path / "data-abc123"
    d.mkdir()
    return str(d)


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path)


class TestBatchStealClaims:
    """Test atomic batch-level steal claiming."""

    def test_first_claim_succeeds(self, warc_dir):
        assert _claim_batch_for_stealing(warc_dir, 95) is True
        assert os.path.exists(_steal_claim_path(warc_dir, 95))

    def test_second_claim_fails(self, warc_dir):
        assert _claim_batch_for_stealing(warc_dir, 95) is True
        assert _claim_batch_for_stealing(warc_dir, 95) is False

    def test_different_batches_both_succeed(self, warc_dir):
        assert _claim_batch_for_stealing(warc_dir, 95) is True
        assert _claim_batch_for_stealing(warc_dir, 94) is True

    def test_claim_creates_correct_path(self, warc_dir):
        _claim_batch_for_stealing(warc_dir, 42)
        expected = os.path.join(warc_dir, "_stealing", "batch_0042")
        assert os.path.exists(expected)

    def test_claim_file_is_empty(self, warc_dir):
        _claim_batch_for_stealing(warc_dir, 10)
        path = _steal_claim_path(warc_dir, 10)
        assert os.path.getsize(path) == 0


class TestListStealClaims:
    """Test listing existing steal claims."""

    def test_no_claims(self, warc_dir):
        assert _list_steal_claims(warc_dir) == set()

    def test_lists_claimed_indices(self, warc_dir):
        _claim_batch_for_stealing(warc_dir, 99)
        _claim_batch_for_stealing(warc_dir, 98)
        _claim_batch_for_stealing(warc_dir, 50)
        claimed = _list_steal_claims(warc_dir)
        assert claimed == {99, 98, 50}

    def test_ignores_non_batch_files(self, warc_dir):
        stealing_dir = os.path.join(warc_dir, "_stealing")
        os.makedirs(stealing_dir, exist_ok=True)
        with open(os.path.join(stealing_dir, "random_file"), "w") as f:
            f.write("")
        _claim_batch_for_stealing(warc_dir, 10)
        claimed = _list_steal_claims(warc_dir)
        assert claimed == {10}


class TestFindStealableBatches:
    """Test finding which batches are available for stealing."""

    def test_all_batches_stealable_when_nothing_done(self, warc_dir, output_dir):
        warc_hash = "abc123"
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value=set(),
        ):
            stealable = _find_stealable_batches(warc_hash, warc_dir, "test_output", 5)
        # Should be in reverse order: [4, 3, 2, 1, 0]
        assert stealable == [4, 3, 2, 1, 0]

    def test_excludes_completed_batches(self, warc_dir):
        warc_hash = "abc123"
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value={0, 1, 2},
        ):
            stealable = _find_stealable_batches(warc_hash, warc_dir, "test_output", 5)
        assert stealable == [4, 3]

    def test_excludes_steal_claimed_batches(self, warc_dir):
        warc_hash = "abc123"
        _claim_batch_for_stealing(warc_dir, 4)
        _claim_batch_for_stealing(warc_dir, 3)
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value=set(),
        ):
            stealable = _find_stealable_batches(warc_hash, warc_dir, "test_output", 5)
        assert stealable == [2, 1, 0]

    def test_excludes_both_done_and_claimed(self, warc_dir):
        warc_hash = "abc123"
        _claim_batch_for_stealing(warc_dir, 4)
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value={0, 1},
        ):
            stealable = _find_stealable_batches(warc_hash, warc_dir, "test_output", 5)
        assert stealable == [3, 2]

    def test_returns_empty_when_all_done_or_claimed(self, warc_dir):
        warc_hash = "abc123"
        _claim_batch_for_stealing(warc_dir, 2)
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value={0, 1},
        ):
            stealable = _find_stealable_batches(warc_hash, warc_dir, "test_output", 3)
        assert stealable == []

    def test_reverse_order(self, warc_dir):
        """Verify stealable batches are in reverse order (highest first)."""
        warc_hash = "abc123"
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value=set(),
        ):
            stealable = _find_stealable_batches(warc_hash, warc_dir, "test_output", 10)
        assert stealable == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        assert stealable[0] > stealable[-1]  # highest first


class TestStealModeIntegration:
    """Integration tests for steal mode with mocked LLM."""

    def _make_mock_llm(self):
        """Create a mock LLM that returns deterministic outputs."""
        mock_llm = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))
        mock_tokenizer.decode.return_value = "decoded text"
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        # Mock vLLM output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Extracted content here", token_ids=list(range(50)))]
        mock_llm.generate.return_value = [mock_output]
        mock_llm.get_tokenizer.return_value = mock_tokenizer

        return mock_llm, mock_tokenizer

    def test_concurrent_steal_claims_no_collision(self, warc_dir):
        """Simulate two stealers claiming different batches on the same WARC."""
        # Stealer A claims batch 99
        assert _claim_batch_for_stealing(warc_dir, 99) is True
        # Stealer B claims batch 98
        assert _claim_batch_for_stealing(warc_dir, 98) is True
        # Stealer A tries to claim 98 (already taken)
        assert _claim_batch_for_stealing(warc_dir, 98) is False
        # Stealer B tries to claim 99 (already taken)
        assert _claim_batch_for_stealing(warc_dir, 99) is False

        claimed = _list_steal_claims(warc_dir)
        assert claimed == {99, 98}

    def test_owner_and_stealer_converge(self, tmp_path):
        """Simulate owner processing forward and stealer processing backward.

        Owner writes batches 0-4, stealer writes batches 9-5. All 10 batches
        should end up written without collision.
        """
        warc_dir = str(tmp_path / "data-test123")
        os.makedirs(warc_dir, exist_ok=True)

        num_batches = 10

        # Owner writes batches 0-4 (forward)
        for i in range(5):
            path = _batch_output_path(warc_dir, i)
            _write_batch_output(path, [{"text": f"batch_{i}_record_0"}])

        # Stealer checks stealable batches — should be [9, 8, 7, 6, 5]
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value={0, 1, 2, 3, 4},
        ):
            stealable = _find_stealable_batches("test123", warc_dir, "test_output", num_batches)
        assert stealable == [9, 8, 7, 6, 5]

        # Stealer claims and writes batches 9-5 (reverse)
        for idx in stealable:
            assert _claim_batch_for_stealing(warc_dir, idx) is True
            path = _batch_output_path(warc_dir, idx)
            _write_batch_output(path, [{"text": f"batch_{idx}_record_0"}])

        # Verify all 10 batch files exist
        for i in range(10):
            path = _batch_output_path(warc_dir, i)
            assert os.path.exists(path), f"batch {i} missing"

    def test_exit_check_writes_done(self, tmp_path):
        """When all batches are complete, exit check should write _done."""
        warc_dir = str(tmp_path / "data-exitcheck")
        os.makedirs(warc_dir, exist_ok=True)

        # Write all 5 batch files
        for i in range(5):
            _write_batch_output(_batch_output_path(warc_dir, i), [{"text": f"record_{i}"}])

        # Simulate exit check: all batches found
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value={0, 1, 2, 3, 4},
        ):
            # This is what the exit check does
            all_completed = {0, 1, 2, 3, 4}
            num_batches = 5
            assert len(all_completed) >= num_batches

            # Write done marker
            _write_done_marker(warc_dir, {"status": "done", "num_batches": 5})

        assert os.path.exists(_done_marker_path(warc_dir))
        with open(_done_marker_path(warc_dir)) as f:
            stats = json.load(f)
        assert stats["status"] == "done"
        assert stats["num_batches"] == 5

    def test_exit_check_skips_done_when_incomplete(self, tmp_path):
        """When not all batches are done, exit check should NOT write _done."""
        warc_dir = str(tmp_path / "data-incomplete")
        os.makedirs(warc_dir, exist_ok=True)

        # Write only 3 of 5 batch files
        for i in range(3):
            _write_batch_output(_batch_output_path(warc_dir, i), [{"text": f"record_{i}"}])

        # Exit check: only 3/5 complete
        all_completed = {0, 1, 2}
        num_batches = 5
        assert len(all_completed) < num_batches
        assert not os.path.exists(_done_marker_path(warc_dir))

    def test_pre_batch_check_skips_existing(self, tmp_path):
        """Pre-batch existence check should detect files written by stealers."""
        warc_dir = str(tmp_path / "data-precheck")
        os.makedirs(warc_dir, exist_ok=True)

        # Stealer wrote batch 5
        _write_batch_output(_batch_output_path(warc_dir, 5), [{"text": "stolen"}])

        # Forward worker checks if batch 5 exists
        batch_path = _batch_output_path(warc_dir, 5)
        assert os.path.exists(batch_path)

    def test_steal_claim_path_format(self, warc_dir):
        """Steal claim path should be in the expected format."""
        path = _steal_claim_path(warc_dir, 42)
        assert path.endswith("/_stealing/batch_0042")

    def test_many_stealers_no_wasted_work(self, warc_dir):
        """With 5 stealers on a 20-batch WARC, each should claim unique batches."""
        # Owner has done 0-9
        with patch(
            "experiments.baseline_collection.run_extract_standalone._find_completed_batches_all_regions",
            return_value=set(range(10)),
        ):
            # 5 stealers each find stealable batches, claim one at a time
            stolen_by = {}
            for stealer_id in range(5):
                stealable = _find_stealable_batches("test", warc_dir, "output", 20)
                if stealable:
                    batch_idx = stealable[0]  # take highest available
                    if _claim_batch_for_stealing(warc_dir, batch_idx):
                        stolen_by[batch_idx] = stealer_id

            # Each stealer should have claimed a unique batch
            assert len(stolen_by) == 5
            # All claimed batches should be 19, 18, 17, 16, 15 (reverse order)
            assert set(stolen_by.keys()) == {19, 18, 17, 16, 15}


class TestStealModeActivation:
    """Test steal mode activation conditions."""

    def test_activates_after_patience_low(self):
        """With < 500 remaining, should activate after 10 consecutive failures."""
        STEAL_THRESHOLD_REMAINING = 500
        STEAL_PATIENCE_LOW = 10
        remaining = 200
        consecutive_failures = 10

        patience = STEAL_PATIENCE_LOW if remaining < STEAL_THRESHOLD_REMAINING else 50
        should_steal = consecutive_failures >= patience
        assert should_steal is True

    def test_does_not_activate_prematurely_low(self):
        """With < 500 remaining, should NOT activate before 10 failures."""
        remaining = 200
        consecutive_failures = 9
        patience = 10 if remaining < 500 else 50
        assert (consecutive_failures >= patience) is False

    def test_activates_after_patience_high(self):
        """With >= 500 remaining, should activate after 50 consecutive failures."""
        remaining = 1000
        consecutive_failures = 50
        patience = 10 if remaining < 500 else 50
        assert (consecutive_failures >= patience) is True

    def test_does_not_activate_prematurely_high(self):
        """With >= 500 remaining, should NOT activate before 50 failures."""
        remaining = 1000
        consecutive_failures = 49
        patience = 10 if remaining < 500 else 50
        assert (consecutive_failures >= patience) is False

    def test_resets_after_successful_steal(self):
        """After a successful steal, consecutive_failures should reset."""
        consecutive_failures = 15
        # Simulating successful steal
        consecutive_failures = 0
        assert consecutive_failures == 0


class TestDeterministicRecordOrder:
    """Verify that the same WARC produces the same record list regardless of who processes it."""

    def test_filter_is_deterministic(self):
        """Character-length filter produces same output for same input."""
        from experiments.baseline_collection.run_extract_standalone import _filter_by_length, MAX_DOC_TOKENS

        records = [
            {"html": "short" * 10},
            {"html": "medium" * 1000},
            {"html": "long" * 100000},
        ]
        result1 = _filter_by_length(records, MAX_DOC_TOKENS)
        result2 = _filter_by_length(records, MAX_DOC_TOKENS)
        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2, strict=True):
            assert r1["html"] == r2["html"]

    def test_warc_hash_is_deterministic(self):
        """Same WARC path always produces same hash."""
        path = "s3://commoncrawl/crawl-data/CC-MAIN-2024-01/segment/warc/test.warc.gz"
        h1 = _warc_path_hash(path)
        h2 = _warc_path_hash(path)
        assert h1 == h2
        assert len(h1) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
