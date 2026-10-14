"""Replay backend — reads saved AutoHDR output fixtures with hash verification."""
import json, hashlib
from pathlib import Path

REPLAY_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "autohdr_replay_synthetic"

def _verify_fixture():
    mf = REPLAY_DIR / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError("Replay fixture manifest not found")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    if manifest.get("fixture_type") != "synthetic_contract_fixture":
        raise ValueError("Invalid fixture type")
    # Verify artifact hashes
    for name, expected_sha in manifest.get("artifacts", {}).items():
        fp = REPLAY_DIR / name
        if not fp.exists():
            raise FileNotFoundError(f"Fixture artifact missing: {name}")
        actual = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected_sha[:16]}..., got {actual[:16]}...")
    return manifest

def replay_detect(image_bytes: bytes = None) -> dict:
    _verify_fixture()
    data = json.loads((REPLAY_DIR / "analysis.json").read_text(encoding="utf-8"))
    data["is_replay"] = True
    data["mock"] = False
    return data

def replay_predict(damages: list = None, ocr_chars: list = None) -> dict:
    data = json.loads((REPLAY_DIR / "analysis.json").read_text(encoding="utf-8"))
    return {"predictions": data.get("damages", []), "is_replay": True}

def replay_restore(image_bytes: bytes = None, confirmed: list = None) -> dict:
    _verify_fixture()
    rp = REPLAY_DIR / "restored.png"
    return {"status": "completed", "restored_path": str(rp) if rp.exists() else None, "is_replay": True}
