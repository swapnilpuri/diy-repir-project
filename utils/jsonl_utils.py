import json
from pathlib import Path
from typing import Any, Iterable

def append_jsonl(path: Path, obj):
    """Append one JSON-serializable object as a UTF-8 JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable):
    """Replace a JSONL file with one serialized record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # Accept Pydantic records as well as dictionaries. model_dump() uses
            # Python mode, so callers must convert datetime values to JSON form.
            if hasattr(record, "model_dump"):
                record = record.model_dump()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    """Load nonblank lines as independent records; JSONDecodeError propagates."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path):
    """Read a single JSON document, including pretty-printed analysis reports."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    """Replace a report with one indented UTF-8 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """
    Load JSONL records from a file.

    Blank lines are ignored.

    Unlike read_jsonl(), decoding errors include the source path and physical
    line number. All records are loaded into memory; no schema check is applied.

    Raises:
        FileNotFoundError:
            If the input file does not exist.

        ValueError:
            If a line is not valid JSON.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        # Count physical lines before skipping blanks so errors match the editor.
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in {file_path}: {exc}"
                ) from exc

            records.append(record)

    return records
