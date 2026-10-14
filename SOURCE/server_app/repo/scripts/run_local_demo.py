"""Start local demo with mock backend."""
import os, subprocess, sys
os.environ.setdefault("AUTOHDR_BACKEND","mock")
os.environ.setdefault("ENABLE_LEGACY_DEBUG","false")
os.environ.setdefault("ENABLE_LEGACY_DIFFHDR","false")
print("Gurou Local Demo — http://127.0.0.1:8000")
print("Backend: mock | AutoHDR: disabled | CBETA: available")
subprocess.run([sys.executable,"-m","uvicorn","demo.backend.main:app","--host","127.0.0.1","--port","8000"])
