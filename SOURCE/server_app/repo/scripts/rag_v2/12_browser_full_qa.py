from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get("RAG_V2_BROWSER_BASE_URL", "http://127.0.0.1:8765")
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "autohdr_replay_synthetic" / "input.png"
REPORT = PROJECT_ROOT / "reports" / "rag_v2" / "10_browser_full_qa.json"
SCREENSHOT = PROJECT_ROOT / "reports" / "rag_v2" / "10_browser_full_qa.png"


def main() -> int:
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(BASE_URL, wait_until="networkidle")

        page.locator('[data-mod="module2"]').click()
        expect(page.locator("#module2")).to_be_visible()
        expect(page.locator("#m2CorpusScope")).to_have_text("全量 CBETA")
        expect(page.locator("#m2SutraCount")).to_have_text("4868")
        expect(page.locator("#m2RagStatus")).to_have_text("RAG V2 稀疏检索")
        expect(page.locator("#m2VectorStatus")).to_have_text("未构建")
        expect(page.locator("#m2BkStatus")).to_have_text("Replay（回放）")

        page.locator("#m2FileInp").set_input_files(str(FIXTURE))
        page.locator("#m2BtnUp").click()
        expect(page.locator("#m2BtnAnalyze")).to_be_visible()
        page.locator("#m2BtnAnalyze").click()
        expect(page.locator("#m2BtnEvidence")).to_be_visible(timeout=30_000)
        page.locator("#m2BtnEvidence").click()
        expect(page.locator(".cbeta-evidence").first).to_be_visible(timeout=60_000)

        citation = page.locator(".cbeta-citation").first.inner_text()
        provenance = page.locator(".cbeta-provenance").first.inner_text()
        details = page.locator(".cbeta-source-window").first
        expect(details).to_be_visible()
        details.locator("summary").click()
        source_text = details.locator(".cbeta-source-text").inner_text()
        source_range = details.locator(".cbeta-source-range").inner_text()
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        print(f"citation={citation}")
        print(f"provenance={provenance}")
        print(f"evidence_range={source_range}")
        if (
            "《" not in citation
            or "卷" not in citation
            or "CBETA Normal Text 2025 R3" not in provenance
            or "文本哈希" not in provenance
        ):
            raise AssertionError("Rendered evidence does not expose the real CBETA source")
        if len(source_text) < 100 or "_p" not in source_range:
            raise AssertionError("Rendered long evidence window is incomplete")
        browser.close()

    if page_errors or console_errors:
        raise AssertionError(
            f"Browser errors: page={page_errors!r}, console={console_errors!r}"
        )
    report = {
        "status": "BROWSER_FULL_RAG_PASS",
        "base_url": BASE_URL,
        "corpus_scope": "全量 CBETA",
        "sutra_count": 4868,
        "vector_status": "未构建",
        "backend_mode": "Replay（回放）",
        "citation": citation,
        "provenance": provenance,
        "evidence_range": source_range,
        "evidence_characters": len(source_text),
        "console_errors": console_errors,
        "page_errors": page_errors,
        "screenshot": str(SCREENSHOT),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
