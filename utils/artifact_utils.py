from __future__ import annotations

from datetime import datetime
from pathlib import Path

# New stage artifacts use the separated UTC timestamp; Step 1 also writes the
# compact form. Both are accepted when locating artifacts from previous runs.
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
ACCEPTED_TIMESTAMP_FORMATS = ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S")


def extract_timestamp(path: Path, prefix: str) -> datetime | None:
    """Parse an artifact's filename timestamp, returning None for unrecognized names."""
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    # Only the text after the exact stage prefix is interpreted as a timestamp.
    value = stem.removeprefix(prefix)
    for fmt in ACCEPTED_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def find_latest_timestamped_file(directory: Path, prefix: str, suffix: str) -> Path:
    """Select the greatest embedded timestamp, independently of file modification time."""
    directory = Path(directory).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # Ignore matching directories and filenames whose timestamp cannot be parsed.
    candidates: list[tuple[datetime, Path]] = []
    for path in directory.glob(f"{prefix}*{suffix}"):
        if path.is_file():
            timestamp = extract_timestamp(path, prefix)
            if timestamp is not None:
                candidates.append((timestamp, path))

    if not candidates:
        raise FileNotFoundError(
            f"No files matching {prefix}<timestamp>{suffix} found in {directory}"
        )
    return max(candidates, key=lambda value: value[0])[1].resolve()


def resolve_input_file(
    cli_input: Path | None,
    directory: Path,
    prefix: str,
    suffix: str = ".jsonl",
) -> Path:
    """Use an explicit CLI input when supplied, otherwise locate the latest stage file."""
    if cli_input is not None:
        # Explicit paths are resolved from cwd; they need not match the stage
        # prefix or live in the default directory, but must have the expected suffix.
        path = Path(cli_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        if suffix and path.suffix.lower() != suffix.lower():
            raise ValueError(f"Expected a {suffix} file: {path}")
        return path

    return find_latest_timestamped_file(directory, prefix, suffix)
