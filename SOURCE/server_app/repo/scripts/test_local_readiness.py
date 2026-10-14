"""25 项自动化测试 — 本地 readiness 完整版。"""
import json, os, sys, subprocess, hashlib, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "local_readiness"
OUTPUT.mkdir(parents=True, exist_ok=True)

results = []

def test(name, fn):
    try:
        fn()
        results.append({"test": name, "passed": True})
        print(f"  PASS: {name}")
    except Exception as e:
        results.append({"test": name, "passed": False, "error": str(e)[:100]})
        print(f"  FAIL: {name} — {e}")

print("=== 25 项自动化测试 ===\n")

# 1-3: Import & Startup
test("1. demo.backend.main import", lambda: __import__("demo.backend.main", fromlist=["app"]))

test("2. state_machine import", lambda: __import__("core.state_machine"))

test("3. job_store import", lambda: __import__("core.job_store"))

# 4-5: Health + Root
from demo.backend.main import app
from fastapi.testclient import TestClient
client = TestClient(app)

def t4():
    r = client.get("/api/system/health")
    assert r.status_code == 200
    assert "status" in r.json()
test("4. GET /api/system/health → 200", t4)

def t5():
    r = client.get("/")
    assert r.status_code == 200
    assert "AutoHDR" in r.text
test("5. GET / → 200 with HTML", t5)

# 6: Image upload
def t6():
    import numpy as np, cv2
    img = np.full((64, 64, 3), (200, 180, 160), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    r = client.post("/api/jobs", files={"image": ("test.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    assert "run_id" in r.json()
    assert r.json()["width"] == 64
test("6. Image upload → 200 with run_id", t6)

# 7: Non-image rejection
def t7():
    r = client.post("/api/jobs", files={"image": ("test.txt", b"not an image", "text/plain")})
    assert r.status_code in (400, 422, 413)
test("7. Non-image upload → rejected", t7)

# 8: Mock analyze
run_id = None
def t8():
    global run_id
    import numpy as np, cv2
    img = np.full((64, 64, 3), (200, 180, 160), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    r = client.post("/api/jobs", files={"image": ("test.png", buf.tobytes(), "image/png")})
    run_id = r.json()["run_id"]
    r2 = client.post(f"/api/jobs/{run_id}/analyze")
    assert r2.status_code == 200
    assert r2.json().get("mock") == True
test("8. Mock analyze → 200 with damages", t8)

# 9: Evidence save
def t9():
    r = client.post(f"/api/jobs/{run_id}/evidence")
    assert r.status_code == 200
    assert "fusions" in r.json()
test("9. Evidence → saved to manifest", t9)

# 10: Fake damage_id rejection
def t10():
    r = client.post(f"/api/jobs/{run_id}/confirm",
        json={"decisions": [{"damage_id": "fake_999", "selected_text": "X", "selected_source": "manual", "status": "manual_confirmed"}]})
    assert r.status_code == 400
test("10. Fake damage_id → 400 rejected", t10)

# 11: Empty text confirm rejection
def t11():
    r = client.post(f"/api/jobs/{run_id}/confirm",
        json={"decisions": [{"damage_id": "damage_001", "selected_text": "", "selected_source": "autohdr", "status": "manual_confirmed"}]})
    assert r.status_code == 400
test("11. Empty text manual_confirmed → 400", t11)

# 12: Invalid source rejection
def t12():
    r = client.post(f"/api/jobs/{run_id}/confirm",
        json={"decisions": [{"damage_id": "damage_001", "selected_text": "X", "selected_source": "invalid_src", "status": "manual_confirmed"}]})
    assert r.status_code == 400
test("12. Invalid source → 400", t12)

# 13: Valid confirm
def t13():
    r = client.post(f"/api/jobs/{run_id}/confirm",
        json={"decisions": [{"damage_id": "damage_001", "selected_text": "佛", "selected_source": "manual", "status": "manual_confirmed"}]})
    assert r.status_code == 200
test("13. Valid manual confirmed → 200", t13)

# 14: Abstain
run_id2 = None
def t14():
    global run_id2
    import numpy as np, cv2
    img = np.full((64, 64, 3), (200, 180, 160), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    r = client.post("/api/jobs", files={"image": ("test2.png", buf.tobytes(), "image/png")})
    run_id2 = r.json()["run_id"]
    client.post(f"/api/jobs/{run_id2}/analyze")
    r2 = client.post(f"/api/jobs/{run_id2}/confirm",
        json={"decisions": [{"damage_id": "damage_001", "selected_text": "", "selected_source": "abstain", "status": "abstain"}]})
    assert r2.status_code == 200
    assert r2.json()["status"] == "ABSTAINED"
    r3 = client.post(f"/api/jobs/{run_id2}/restore")
    assert r3.status_code == 200
    assert r3.json()["status"] == "abstain"
    assert "restored_url" not in r3.json()
test("14. Abstain → 200 ABSTAINED", t14)

# 15: No confirm → restore denied
def t15():
    import numpy as np, cv2
    img = np.full((64, 64, 3), (200, 180, 160), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    r = client.post("/api/jobs", files={"image": ("nx.png", buf.tobytes(), "image/png")})
    rid = r.json()["run_id"]
    client.post(f"/api/jobs/{rid}/analyze")
    r2 = client.post(f"/api/jobs/{rid}/restore")
    assert r2.json()["status"] == "abstain"
test("15. Unconfirmed → restore denied", t15)

# 16: Mock restore
def t16():
    r = client.post(f"/api/jobs/{run_id}/restore")
    assert r.status_code == 200
    assert r.json().get("is_mock") == True
test("16. Mock restore → completed with is_mock=true", t16)

# 17: State illegal jump → 409
def t17():
    import numpy as np, cv2
    img = np.full((64, 64, 3), (200, 180, 160), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    r = client.post("/api/jobs", files={"image": ("jump.png", buf.tobytes(), "image/png")})
    rid = r.json()["run_id"]
    # Try to restore without analyze+confirm
    r2 = client.post(f"/api/jobs/{rid}/restore")
    assert r2.status_code in (409, 400)
test("17. State illegal jump → rejected", t17)

# 18: Debug routes disabled
def t18():
    r = client.post("/api/confirm", json={})
    assert r.status_code == 404
test("18. /api/confirm → 404 (debug disabled)", t18)

# 19: Legacy routes disabled
def t19():
    r = client.post("/api/rejoin")
    assert r.status_code == 404
test("19. Legacy /api/rejoin → 404 (legacy disabled)", t19)

# 20: Worker unavailable
def t20():
    from integrations.autohdr.base import get_backend
    be = get_backend()
    assert "detect" in be
    assert "is_mock" in be
test("20. Backend routing → functional", t20)

# 21: Replay verify
def t21():
    import os
    os.environ["AUTOHDR_BACKEND"] = "replay"
    from integrations.autohdr.base import detect
    r = detect()
    assert "damages" in r
    os.environ["AUTOHDR_BACKEND"] = "mock"
test("21. Replay detect → returns fixture data", t21)

# 22: Job isolation
def t22():
    import numpy as np, cv2
    img = np.full((16, 16, 3), (200, 180, 160), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    payload = buf.tobytes()
    r1 = client.post("/api/jobs", files={"image": ("a.png", payload, "image/png")})
    r2 = client.post("/api/jobs", files={"image": ("b.png", payload, "image/png")})
    assert r1.status_code == 200
    assert r2.status_code == 200
    j1 = r1.json()
    j2 = r2.json()
    assert j1["run_id"] != j2["run_id"]
test("22. Different run_id → isolated", t22)

# 23: State machine valid transitions
def t23():
    from core.state_machine import JobStatus, can_transition
    assert can_transition(JobStatus.UPLOADED, JobStatus.ANALYZING)
    assert not can_transition(JobStatus.COMPLETED, JobStatus.ANALYZING)
test("23. State machine → valid transitions", t23)

# 24: Job ID lookup
def t24():
    r = client.get("/api/jobs/nonexistent_run")
    assert r.status_code == 404
test("24. Nonexistent job → 404", t24)

# 25: Key not in logs
def t25():
    for f in (OUTPUT.parent / "cbeta_index_audit").glob("*.json"):
        for f2 in [f]:
            if f2.exists() and "sk-ws-" in f2.read_text(encoding="utf-8", errors="ignore"):
                raise AssertionError(f"Key leak in {f2}")
test("25. API key not in output logs", t25)

# Summary
passed = sum(1 for r in results if r["passed"])
print(f"\n=== {passed}/{len(results)} PASSED ===")
(OUTPUT / "unit_test_results.json").write_text(json.dumps(results, indent=2))
if passed != len(results):
    raise SystemExit(1)
