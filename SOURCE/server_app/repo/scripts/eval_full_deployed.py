"""Run the frozen 570-case evaluator against an installed RAG V2 release."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    script_root = project_root / "scripts" / "rag_v2"
    sys.path.insert(0, str(script_root))
    sys.path.insert(0, str(project_root))

    import common  # type: ignore[import-not-found]

    common.FULL_ARTIFACT = args.runtime.resolve()
    common.FULL_DB = common.FULL_ARTIFACT / "metadata.sqlite3"
    common.REPORT_ROOT = args.report_root.resolve()
    common.CACHE_ROOT = args.cache_root.resolve()
    if not common.FULL_DB.is_file():
        raise FileNotFoundError(common.FULL_DB)
    sys.argv = [str(script_root / "08_eval_full.py")]
    runpy.run_path(str(script_root / "08_eval_full.py"), run_name="__main__")


if __name__ == "__main__":
    main()
