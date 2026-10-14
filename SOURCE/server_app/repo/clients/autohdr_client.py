"""AutoHDR 客户端 — 主项目通过 HTTP 调用 Worker。"""
import requests
import os
from typing import Optional

WORKER_URL = os.environ.get("AUTOHDR_SERVICE_URL", "http://127.0.0.1:8010")
TIMEOUT = 300


class AutoHDRClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or WORKER_URL

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.json()
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

    def analyze(self, image_bytes: bytes, manual_mask: Optional[bytes] = None) -> dict:
        files = {"image": image_bytes}
        if manual_mask:
            files["mask"] = manual_mask
        r = requests.post(f"{self.base_url}/analyze", files=files, timeout=TIMEOUT)
        return r.json()

    def restore(self, run_id: str, confirmed_characters: dict) -> dict:
        r = requests.post(f"{self.base_url}/restore", json={
            "run_id": run_id,
            "confirmed_characters": confirmed_characters,
        }, timeout=TIMEOUT)
        return r.json()

    def get_run(self, run_id: str) -> dict:
        r = requests.get(f"{self.base_url}/runs/{run_id}", timeout=10)
        return r.json()

    def is_available(self) -> bool:
        return self.health().get("status") == "ok"
