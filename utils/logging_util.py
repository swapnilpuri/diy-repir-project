import json

from datetime import datetime, timezone
from pathlib import Path

def log_event(event_type: str, event_data: dict, log_dir: Path):
    """
    Append an event with a UTC timestamp to the event type's log file.

    Despite its .json suffix, the file contains independent JSON objects, one
    per line (JSONL), rather than a single JSON document or array. The main
    pipeline does not currently call this helper.

    Args:
        event_type (str): The type of the event (e.g., "info", "error").
        event_data (dict): A dictionary containing the event data.
        log_dir (Path): The directory where the log file will be stored.
    """
    # Allow the first event to create its destination directory.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Record an ISO 8601 timestamp with an explicit UTC offset.
    timestamp = datetime.now(timezone.utc).isoformat()

    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "event_data": event_data
    }

    # Separate event types into files; event_type should be a filename-safe label.
    log_file_path = log_dir / f"{event_type}_log.json"

    # Append preserves earlier events; writes are not coordinated across callers.
    with open(log_file_path, "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")  # Write a newline for readability
