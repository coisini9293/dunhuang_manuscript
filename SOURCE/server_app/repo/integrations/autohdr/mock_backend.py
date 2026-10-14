"""Mock AutoHDR backend — dynamic bbox based on image size."""
import cv2, numpy as np

def mock_detect(image_bytes=None):
    w, h = 512, 512
    if image_bytes:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
    damages = [{
        "damage_id": "damage_001",
        "bbox": [int(w*0.45), int(h*0.40), max(20,int(w*0.08)), max(20,int(h*0.12))],
        "left_context": "如是我聞一時佛在摩竭提國",
        "right_context": "寂滅道場始成正覺其地金剛",
        "autohdr_candidates": [{"text":"寂","rank":1,"score":0.85},{"text":"滅","rank":2,"score":0.72}]
    }]
    chars = [
        {"char_id":"char_001","bbox":[int(w*0.2),int(h*0.3),30,40],"text":"如","reading_order":0},
        {"char_id":"char_002","bbox":[int(w*0.3),int(h*0.3),30,40],"text":"是","reading_order":1},
        {"char_id":"char_003","bbox":[int(w*0.45),int(h*0.4),30,40],"text":"","reading_order":2},
    ]
    return {"damages":damages,"ocr_chars":chars,"mock":True,"backend":"mock","width":w,"height":h}

def mock_predict(damages=None, ocr_chars=None):
    return {"predictions":{},"mock":True}

def mock_restore(image_bytes=None, confirmed=None):
    img = None
    if image_bytes:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((256,128,3),(240,235,220),dtype=np.uint8)
    cv2.putText(img,"MOCK RESTORATION",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(img,"NOT AUTOHDR OUTPUT",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    return {"status":"completed","is_mock":True}
