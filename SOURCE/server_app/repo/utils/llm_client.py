"""LLM client: 通义千问 qwen-max 封装，带缓存/重试/降级 + LLM_ENABLED 开关 + mock 模式。

LLM_ENABLED=false（默认）时返回 mock L3 结果，不调真实 API。
LLM_ENABLED=true 且 DASHSCOPE_API_KEY 以 sk- 开头时才调真实 API。

用于 RAG agent 的 L3 fallback：当 CBETA 本地检索和同文异本库都未命中时，
调用 LLM 基于上下文预测缺失字。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-max")
LLM_ENABLED = os.environ.get("LLM_ENABLED", "false").lower() == "true"

_CACHE: dict[str, dict] = {}
_CACHE_DIR = _PROJECT_ROOT / "data" / "llm_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _load_disk_cache(key: str) -> Optional[dict]:
    p = _CACHE_DIR / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_disk_cache(key: str, data: dict) -> None:
    p = _CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_api_ready() -> bool:
    """检查 LLM 是否真实可用（LLM_ENABLED=true 且 key 格式正确）。"""
    if not LLM_ENABLED:
        return False
    if not DASHSCOPE_API_KEY:
        print("[llm_client] DASHSCOPE_API_KEY detected: false")
        return False
    if not DASHSCOPE_API_KEY.startswith("sk-"):
        print("[llm_client] DASHSCOPE_API_KEY detected: true (format warning)")
        return False
    print("[llm_client] DASHSCOPE_API_KEY detected: true")
    return True


def is_mock_mode() -> bool:
    """是否处于 mock 模式（LLM_ENABLED=false）。"""
    return not LLM_ENABLED


def _mock_predict(
    context_before: str,
    context_after: str,
    sutra_name: str,
    top_k: int,
) -> dict:
    """Mock L3：返回占位结果，标记 source=mock_llm，不调真实 API。

    用于 LLM_ENABLED=false 时跑通 L3 流程，confidence=0.0 强制走 L4 标灰。
    """
    return {
        "candidates": [
            {
                "char": "?",
                "confidence": 0.0,
                "note": "mock L3 (LLM_ENABLED=false)",
            }
        ],
        "raw_response": "mock",
        "source": "mock_llm",
        "mock": True,
    }


def predict_missing_char(
    context_before: str,
    context_after: str,
    sutra_name: str = "",
    top_k: int = 5,
    max_retries: int = 3,
) -> Optional[dict]:
    """基于上下文预测缺失字（L3 fallback）。

    LLM_ENABLED=false 时返回 mock 结果（source=mock_llm）。
    LLM_ENABLED=true 但 API 不可用时返回 None（触发 L4 标灰）。
    LLM_ENABLED=true 且 API 可用时调真实通义千问。
    """
    # Mock 模式：返回占位结果
    if is_mock_mode():
        print("[llm_client] mock L3 模式（LLM_ENABLED=false），返回占位结果")
        return _mock_predict(context_before, context_after, sutra_name, top_k)

    # 真实模式但 API 未就绪：返回 None 触发 L4
    if not is_api_ready():
        print("[llm_client] LLM_ENABLED=true 但 API 未就绪，L3 降级")
        return None

    prompt = _build_prompt(context_before, context_after, sutra_name, top_k)
    key = _cache_key(prompt)

    if key in _CACHE:
        return _CACHE[key]

    disk = _load_disk_cache(key)
    if disk is not None:
        _CACHE[key] = disk
        return disk

    try:
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY
    except ImportError:
        print("[llm_client] dashscope 未安装，L3 降级")
        return None

    for attempt in range(max_retries):
        try:
            response = dashscope.Generation.call(
                model=DASHSCOPE_MODEL,
                prompt=prompt,
                result_format="message",
                temperature=0.3,
                top_p=0.8,
            )
            if response.status_code == 200:
                result = _parse_response(response, top_k)
                _CACHE[key] = result
                _save_disk_cache(key, result)
                return result
            else:
                print(f"[llm_client] API 错误 (attempt {attempt+1}): code={response.status_code} msg={response.message}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[llm_client] 异常 (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print("[llm_client] L3 降级（API 重试耗尽）")
    return None


def _build_prompt(context_before: str, context_after: str, sutra_name: str, top_k: int) -> str:
    sutra_hint = f"（出自{sutra_name}）" if sutra_name else ""
    before_ctx = context_before[-30:] if len(context_before) > 30 else context_before
    after_ctx = context_after[:30] if len(context_after) > 30 else context_after
    return f"""你是一位古汉语文献校勘专家{sutra_hint}。下面是一段敦煌写本的文字转录，其中有一处缺损需要补字。

缺损前文字：...{before_ctx}
[此处缺损]
缺损后文字：{after_ctx}...

请基于上下文和佛经用语习惯，预测缺损处最可能的 {top_k} 个候选字，按可能性从高到低排列。
只输出 JSON，格式：{{"candidates": [{{"char": "字", "confidence": 0.9}}]}}
confidence 取 0-1 之间，反映你对这个字的把握。
如果上下文不足以推断，返回空候选列表：{{"candidates": []}}"""


def _parse_response(response, top_k: int) -> dict:
    try:
        text = response.output.choices[0].message.content
        m = re.search(r'\{[^{}]*"candidates"[^{}]*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            candidates = data.get("candidates", [])[:top_k]
            return {"candidates": candidates, "raw_response": text, "source": "llm_qwen_max"}
        data = json.loads(text)
        candidates = data.get("candidates", [])[:top_k]
        return {"candidates": candidates, "raw_response": text, "source": "llm_qwen_max"}
    except Exception as e:
        print(f"[llm_client] 解析失败: {e}")
    return {"candidates": [], "raw_response": "", "source": "llm_qwen_max"}


def test_connection() -> bool:
    """测试 API 连通性（仅 LLM_ENABLED=true 时有效）。"""
    if not is_api_ready():
        print("[llm_client] API 未就绪（LLM_ENABLED=false 或 key 格式异常）")
        return False
    try:
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY
        response = dashscope.Generation.call(
            model=DASHSCOPE_MODEL,
            prompt="请回复'连通测试成功'四个字",
            result_format="message",
        )
        if response.status_code == 200:
            text = response.output.choices[0].message.content
            print(f"[llm_client] 连通测试成功，模型回复: {text[:50]}")
            return True
        else:
            print(f"[llm_client] 连通测试失败: code={response.status_code} msg={response.message}")
            return False
    except Exception as e:
        print(f"[llm_client] 连通测试异常: {e}")
        return False


if __name__ == "__main__":
    print(f"LLM_ENABLED: {LLM_ENABLED}")
    print(f"DASHSCOPE_API_KEY detected: {bool(DASHSCOPE_API_KEY)}")
    print(f"Model: {DASHSCOPE_MODEL}")
    print(f"Mock mode: {is_mock_mode()}")
    print(f"API ready: {is_api_ready()}")
    print()
    # 测试 mock 模式
    print("--- 测试 mock L3 ---")
    result = predict_missing_char("如是我闻", "一时佛在", sutra_name="大方广佛华严经")
    print(f"Result source: {result.get('source')}")
    print(f"Result mock: {result.get('mock', False)}")
