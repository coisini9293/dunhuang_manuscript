"""AutoHDR Backend 路由：根据 AUTOHDR_BACKEND 环境变量选择。"""
import os

BACKEND = os.environ.get("AUTOHDR_BACKEND", "mock")


def get_backend():
    if BACKEND == "mock":
        from integrations.autohdr.mock_backend import mock_detect, mock_predict, mock_restore
        return {"detect": mock_detect, "predict": mock_predict, "restore": mock_restore, "is_mock": True}
    elif BACKEND == "replay":
        from integrations.autohdr.replay_backend import replay_detect, replay_predict, replay_restore
        return {"detect": replay_detect, "predict": replay_predict, "restore": replay_restore, "is_mock": False}
    elif BACKEND == "official":
        from integrations.autohdr.official_subprocess_backend import official_detect, official_predict, official_restore
        return {"detect": official_detect, "predict": official_predict, "restore": official_restore, "is_mock": False}
    else:
        raise ValueError(f"Unknown AUTOHDR_BACKEND: {BACKEND}")


def detect(image_bytes: bytes = None) -> dict:
    return get_backend()["detect"](image_bytes)


def predict(damages: list, ocr_chars: list) -> dict:
    return get_backend()["predict"](damages, ocr_chars)


def restore(image_bytes: bytes = None, confirmed: list = None) -> dict:
    return get_backend()["restore"](image_bytes, confirmed)


def is_mock() -> bool:
    return get_backend()["is_mock"]
