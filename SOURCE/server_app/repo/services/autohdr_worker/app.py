"""AutoHDR Worker — 独立服务，运行 AutoHDR 损伤检测/OCR/文字预测/外观恢复。

使用独立 conda 环境 (autohdr)，默认端口 8010。
主项目通过 HTTP API 调用，不直接 import AutoHDR 依赖。
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AutoHDR Worker")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": "N/A",
        "models_loaded": False,
        "text_model": "not_configured",
        "gpu_memory": {},
        "official_commit": "pending_clone",
        "note": "autohdr_inference_not_executed — no CUDA on this machine",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
