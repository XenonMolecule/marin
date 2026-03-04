# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download files from URLs to GCS.

Generic utility for downloading model binaries and other resources from public
URLs and storing them on GCS for use by pipeline workers.

Usage as an ExecutorStep::

    download_step = ExecutorStep(
        name="resources/my_model",
        fn=download_url_to_gcs,
        config=DownloadUrlToGcsConfig(
            url="https://example.com/model.bin",
            output_path=this_output_path(),
        ),
        resources=ResourceConfig.with_cpu(cpu=2, ram="4g"),
        pip_dependency_groups=["cpu"],
    )
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field

import fsspec
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DownloadUrlToGcsConfig:
    url: str
    """URL to download from."""

    output_path: str
    """GCS directory to write the downloaded file to."""

    filename: str | None = None
    """Override filename. If None, inferred from the URL."""

    http_timeout: int = 600
    """Timeout in seconds for the download."""


@dataclass
class DownloadBanlistsConfig:
    output_path: str
    """GCS directory to write the ban list files to."""

    github_base_url: str = (
        "https://raw.githubusercontent.com/mlfoundations/dclm/main/baselines/mappers"
    )
    """Base URL for DCLM GitHub raw files."""

    files: list[str] = field(
        default_factory=lambda: [
            "banlists/refinedweb_banned_domains_curated.txt",
            "banlists/refinedweb_banned_words_strict_reverse_engineered.txt",
            "banlists/refinedweb_banned_words_hard_reverse_engineered.txt",
            "banlists/refinedweb_banned_words_soft_reverse_engineered.txt",
            "iana_tlds.txt",
        ]
    )
    """Files to download relative to the github_base_url."""


def download_url_to_gcs(config: DownloadUrlToGcsConfig) -> None:
    """Download a file from a URL and upload to GCS.

    Idempotent: skips download if the output file already exists on GCS.
    """
    filename = config.filename or os.path.basename(config.url.split("?")[0])
    output_file = os.path.join(config.output_path, filename)

    fs, _, _ = fsspec.get_fs_token_paths(output_file)
    if fs.exists(output_file):
        logger.info("Output already exists, skipping download: %s", output_file)
        return

    logger.info("Downloading %s -> %s", config.url, output_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
        tmp_path = tmp.name
        response = requests.get(config.url, stream=True, timeout=config.http_timeout)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                tmp.write(chunk)
                pbar.update(len(chunk))

    logger.info("Uploading %s to %s", tmp_path, output_file)
    fs.put(tmp_path, output_file)
    os.unlink(tmp_path)
    logger.info("Download complete: %s", output_file)


def download_dclm_banlists(config: DownloadBanlistsConfig) -> None:
    """Download DCLM ban list files and TLD list from GitHub to GCS.

    Downloads each file from the DCLM GitHub repository and uploads to GCS.
    Idempotent: skips files that already exist.
    """
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)

    for relative_path in config.files:
        url = f"{config.github_base_url}/{relative_path}"
        output_file = os.path.join(config.output_path, os.path.basename(relative_path))

        if fs.exists(output_file):
            logger.info("Already exists, skipping: %s", output_file)
            continue

        logger.info("Downloading %s -> %s", url, output_file)
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with fs.open(output_file, "wb") as f:
            f.write(response.content)

        logger.info("Downloaded %s (%d bytes)", output_file, len(response.content))

    logger.info("All ban lists downloaded to %s", config.output_path)
