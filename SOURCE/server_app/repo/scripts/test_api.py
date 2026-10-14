"""阶段 0：API 连通性测试。

测试通义千问 Qwen API 连通性，输出结构化结果。
不记录 Authorization Header，不打印完整 Key。

输出：
  outputs/api_test/
    api_test_result.json  (provider, model, success, latency, token_usage, error_type, timestamp)
    api_test.log          (完整推理日志，不含 Key)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "api_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / "api_test.log"
RESULT_PATH = OUTPUT_DIR / "api_test_result.json"


def log(msg: str):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_api_test() -> dict:
    """执行 API 连通性测试。"""
    # 清空日志
    LOG_PATH.write_text("", encoding="utf-8")

    log("=" * 60)
    log("阶段 0：API 连通性测试")
    log(f"时间: {datetime.now(timezone.utc).isoformat()}")
    log("=" * 60)

    import os
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-max")
    llm_enabled = os.environ.get("LLM_ENABLED", "false").lower() == "true"

    log(f"DASHSCOPE_API_KEY detected: {bool(api_key)}")
    log(f"LLM_ENABLED: {llm_enabled}")
    log(f"Model: {model}")
    log(f"Provider: dashscope (阿里云百炼)")

    result = {
        "provider": "dashscope",
        "model": model,
        "success": False,
        "latency_ms": 0,
        "token_usage": None,
        "error_type": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not api_key:
        result["error_type"] = "key_missing"
        log("[FAIL] API Key 未配置")
        _save(result)
        return result

    if not llm_enabled:
        result["error_type"] = "llm_disabled"
        log("[FAIL] LLM_ENABLED=false")
        _save(result)
        return result

    try:
        import dashscope
        dashscope.api_key = api_key
        log("[INFO] dashscope SDK 已加载")
        log("[INFO] 发送最小测试请求（不含真实残卷信息）...")

        test_prompt = "请回复'连通测试成功'六个字，不要输出其他内容。"
        t0 = time.time()
        response = dashscope.Generation.call(
            model=model,
            prompt=test_prompt,
            result_format="message",
            temperature=0.0,
            max_tokens=20,
        )
        latency_ms = int((time.time() - t0) * 1000)
        result["latency_ms"] = latency_ms

        log(f"[INFO] HTTP 状态码: {response.status_code}")
        log(f"[INFO] 耗时: {latency_ms} ms")

        if response.status_code == 200:
            text = response.output.choices[0].message.content
            log(f"[INFO] 模型回复: {text[:80]}")

            # token 使用量
            usage = response.usage
            if usage:
                token_usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                result["token_usage"] = token_usage
                log(f"[INFO] Token 使用: input={token_usage['input_tokens']}, "
                    f"output={token_usage['output_tokens']}, "
                    f"total={token_usage['total_tokens']}")

            result["success"] = True
            log("[PASS] API 连通性测试成功")
        else:
            result["error_type"] = f"http_{response.status_code}"
            log(f"[FAIL] API 错误: code={response.status_code}")
            log(f"[FAIL] 错误码: {getattr(response, 'code', 'N/A')}")
            log(f"[FAIL] 错误信息: {getattr(response, 'message', 'N/A')}")

    except Exception as e:
        result["error_type"] = type(e).__name__
        log(f"[FAIL] 异常: {type(e).__name__}: {e}")

    _save(result)
    return result


def _save(result: dict):
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log(f"\n[OUTPUT] 结果已保存: {RESULT_PATH}")
    log(f"[OUTPUT] 日志已保存: {LOG_PATH}")


if __name__ == "__main__":
    run_api_test()
