# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheme-preserving fsspec path helpers for fork experiment code.

Upstream marin.utils dropped these in favor of rigging.filesystem.StoragePath;
the fork's experiment scripts predate that idiom and keep the original thin
helpers here (fork-owned) instead of rewriting ~65 call sites.
"""

import braceexpand
import fsspec
from rigging.filesystem import url_to_fs


def fsspec_exists(file_path):
    """
    Check if a file exists in a fsspec filesystem.

    Args:
        file_path (str): The path of the file

    Returns:
        bool: True if the file exists, False otherwise.
    """

    # Use fsspec to check if the file exists
    fs = url_to_fs(file_path)[0]
    return fs.exists(file_path)


def fsspec_glob(file_path):
    """
    Get a list of files in a fsspec filesystem that match a pattern.

    We extend fsspec glob to also work with braces, using braceexpand.

    Args:
        file_path (str): a file path or pattern, possibly with *, **, ?, or {}'s

    Returns:
        list: A list of files that match the pattern. returned files have the protocol prepended to them.
    """

    # Use fsspec to get a list of files
    fs = url_to_fs(file_path)[0]
    protocol = fsspec.core.split_protocol(file_path)[0]

    def join_protocol(file):
        if protocol:
            return f"{protocol}://{file}"
        return file

    out = []

    # glob has to come after braceexpand
    for file in braceexpand.braceexpand(file_path):
        out.extend(join_protocol(file) for file in fs.glob(file))

    return out


def fsspec_size(file_path: str) -> int:
    """Get file size (in bytes) of a file on an `fsspec` filesystem."""
    fs = url_to_fs(file_path)[0]

    return fs.size(file_path)
