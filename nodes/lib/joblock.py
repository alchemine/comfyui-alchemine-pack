"""Shared single-file job lock for fire-and-forget remote jobs.

One `jobs.lock` file in the pack directory tracks at most one in-flight job per
*kind* (e.g. `"api"`, `"grok"`). Each kind is an independent slot: an API job and
a Grok job can be in flight at the same time, but only one of each. The file
persists across restarts, so a mid-flight job is remembered even if ComfyUI is
restarted. A process-level lock guards the read-modify-write so concurrent
submits can't both pass the "is this kind free?" check at the same instant.

File shape::

    {"api": {<record>}, "grok": {<record>}}

A missing key, or a record without a truthy `prompt_id`/`request_id`, means that
kind's slot is free.
"""

import json
import os
import threading

# `nodes/lib/joblock.py` -> pack root is three levels up.
_LOCK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "jobs.lock"
)

# A single guard shared by all kinds: writes touch the same file, so the whole
# read-modify-write must be serialized regardless of kind.
guard = threading.Lock()

# A record is considered "present" if it carries any of these id fields.
_ID_KEYS = ("prompt_id", "request_id", "job_id")


def _read_all() -> dict:
    try:
        with open(_LOCK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_lock(kind: str) -> "dict | None":
    """Return the in-flight record for `kind`, or None if that slot is free."""
    rec = _read_all().get(kind)
    if isinstance(rec, dict) and any(rec.get(k) for k in _ID_KEYS):
        return rec
    return None


def write_lock(kind: str, record: "dict | None") -> None:
    """Atomically set (or clear, when `record` is falsy) the `kind` slot,
    preserving the records of every other kind."""
    data = _read_all()
    if record:
        data[kind] = record
    else:
        data.pop(kind, None)
    tmp = _LOCK_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, _LOCK_FILE)
