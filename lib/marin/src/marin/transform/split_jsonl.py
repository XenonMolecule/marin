# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split large JSONL.gz files into smaller chunks for downstream parallelism.

When a dataset has few large files (e.g. 2 WARC extractions with 50k records each),
downstream Zephyr pipelines only get 2 shards and can't parallelize effectively.
This step splits each file into sub-files of ``records_per_file`` records, giving
Zephyr many more shards to distribute across workers.

Example usage as an executor step::

    split = ExecutorStep(
        name="split/my_dataset",
        fn=split_jsonl_files,
        config=SplitJsonlConfig(
            input_path=raw_data / "*.jsonl.gz",
            output_path=this_output_path(),
            records_per_file=5000,
        ),
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
        pip_dependency_groups=["cpu"],
    )
"""

import dataclasses
import gzip
import logging
import os

import draccus
import fsspec

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SplitJsonlConfig:
    """Configuration for splitting JSONL.gz files into smaller chunks.

    Attributes:
        input_path: Glob pattern for input JSONL.gz files.
        output_path: Directory to write chunked output files.
        records_per_file: Maximum number of records per output file.
    """

    input_path: str
    output_path: str
    records_per_file: int = 5000


def _write_chunk(lines: list[str], output_path: str, stem: str, chunk_idx: int):
    """Write a chunk of JSONL lines to a gzipped output file."""
    output_file = f"{output_path}/{stem}_chunk{chunk_idx:04d}.jsonl.gz"
    with fsspec.open(output_file, "wb") as f_out:
        with gzip.open(f_out, "wt", encoding="utf-8") as gz_out:
            for line in lines:
                gz_out.write(line)


def split_jsonl_files(config: SplitJsonlConfig):
    """Split large JSONL.gz files into smaller chunks."""
    input_files = fsspec_glob(config.input_path)
    if not input_files:
        raise FileNotFoundError(f"No files found matching: {config.input_path}")

    logger.info(f"Splitting {len(input_files)} file(s) into chunks of {config.records_per_file} records")

    total_records = 0
    total_output_files = 0

    for input_file in input_files:
        stem = os.path.basename(input_file).replace(".jsonl.gz", "")
        file_idx = 0
        buffer: list[str] = []

        with fsspec.open(input_file, "rb") as f_in:
            with gzip.open(f_in, "rt", encoding="utf-8") as gz_in:
                for line in gz_in:
                    buffer.append(line)
                    if len(buffer) >= config.records_per_file:
                        _write_chunk(buffer, config.output_path, stem, file_idx)
                        total_records += len(buffer)
                        total_output_files += 1
                        buffer = []
                        file_idx += 1

                if buffer:
                    _write_chunk(buffer, config.output_path, stem, file_idx)
                    total_records += len(buffer)
                    total_output_files += 1

        logger.info(f"Split {os.path.basename(input_file)} into {file_idx + 1} chunks")

    logger.info(f"Split complete: {total_records} records across {total_output_files} files")


@draccus.wrap()
def main(config: SplitJsonlConfig):
    split_jsonl_files(config)


if __name__ == "__main__":
    main()
