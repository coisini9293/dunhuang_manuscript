from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT, REPORT_ROOT, disk_usage, now_iso, write_json


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> int:
    head = git("rev-parse", "HEAD")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "60352fe", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    usage = disk_usage()
    payload = {
        "captured_at": now_iso(),
        "head": head,
        "branch": git("branch", "--show-current"),
        "head_is_60352fe_or_descendant": ancestry,
        "disk": usage,
        "python": sys.version,
        "status": "PASS" if ancestry and usage["free_gib"] >= 8 else "FAIL",
    }
    write_json(REPORT_ROOT / "preflight_runtime.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
