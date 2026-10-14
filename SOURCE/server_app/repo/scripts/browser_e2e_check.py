"""Browser E2E check — 9-step workflow via HTTP."""
import sys,os,cv2,numpy as np
sys.path.insert(0,'.')
from fastapi.testclient import TestClient
from demo.backend.main import app
client=TestClient(app)
passed=0
img=np.full((128,128,3),(200,180,160),dtype=np.uint8)
cv2.putText(img,"TEST",(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
_,buf=cv2.imencode(".png",img)
bd=buf.tobytes()
rid=client.post("/api/jobs",files={"image":("test.png",bd,"image/png")}).json()["run_id"];print(f"1.Upload:{rid}");passed+=1
r=client.post(f"/api/jobs/{rid}/analyze");assert r.status_code==200;print(f"2.Analyze:{len(r.json().get('damages',[]))}dmg");passed+=1
r=client.post(f"/api/jobs/{rid}/evidence");print(f"3.Evidence:{len(r.json().get('fusions',[]))}");passed+=1
r=client.post(f"/api/jobs/{rid}/confirm",json={"decisions":[{"damage_id":"damage_001","selected_text":"佛","selected_source":"manual","status":"manual_confirmed"}]});assert r.status_code==200;print(f"4.Confirm");passed+=1
r=client.post(f"/api/jobs/{rid}/restore");assert r.json().get("is_mock");print(f"5.Restore mock");passed+=1
r=client.get(f"/api/jobs/{rid}/report");assert r.status_code==200;print(f"6.Report");passed+=1
r=client.get(f"/api/jobs/{rid}/download");assert r.status_code==200;print(f"7.Download");passed+=1
rid2=client.post("/api/jobs",files={"image":("t2.png",bd,"image/png")}).json()["run_id"];client.post(f"/api/jobs/{rid2}/analyze")
r=client.post(f"/api/jobs/{rid2}/confirm",json={"decisions":[{"damage_id":"fake_999","selected_text":"X","selected_source":"manual","status":"manual_confirmed"}]});assert r.status_code==400;print(f"8.Fake reject");passed+=1
rid3=client.post("/api/jobs",files={"image":("t3.png",bd,"image/png")}).json()["run_id"];client.post(f"/api/jobs/{rid3}/analyze")
r=client.post(f"/api/jobs/{rid3}/restore");assert r.json().get("status")=="abstain";print(f"9.Unconfirmed deny");passed+=1
print(f"{passed}/9 PASSED")
