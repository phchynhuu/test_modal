"""
Hashing utilities.

`compute_file_hash` is used by workers to hash source media bytes.
This is the second dedup layer: the first (in cache_service.py) hashes
request *parameters*, while this hashes actual *file content*.

If two requests have identical parameters but different source files they
will share a parameter hash — the file-content hash provides the correct
disambiguation when that matters (e.g. two different images both submitted
for upscaling with prompt="").
"""

import hashlib


def compute_file_hash(data: bytes) -> str:
    """Return hex SHA-256 of raw bytes.  Fast enough for files up to ~500 MB."""
    return hashlib.sha256(data).hexdigest()


def compute_combined_hash(file_hash: str, params: dict) -> str:
    """
    Combine a file content hash with operation parameters into a single
    dedup key.  Used when both file identity and params matter.
    """
    import json
    combined = json.dumps(
        {"file": file_hash, "params": params},
        sort_keys=True,
    )
    return hashlib.sha256(combined.encode()).hexdigest()
