from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .common import canonical_sha256


def append_audit(path: Path, action: str, payload: dict) -> dict:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "payload_sha256": canonical_sha256(payload),
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event

