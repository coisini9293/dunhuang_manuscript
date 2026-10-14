from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from common import FULL_ARTIFACT, PROJECT_ROOT, disk_usage, now_iso, read_json, write_json


DEPLOY_ROOT = PROJECT_ROOT / "artifacts" / "rag_v2_deploy"
MODEL_SOURCE = PROJECT_ROOT / "models" / "embeddings"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "hardlink"
        except OSError:
            pass
        if (
            destination.stat().st_size == source.stat().st_size
            and sha256_file(destination) == sha256_file(source)
        ):
            return "existing-verified"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    index_manifest_path = FULL_ARTIFACT / "manifests" / "index_manifest.json"
    if not index_manifest_path.exists():
        raise RuntimeError("Full index has not passed final validation")
    index_manifest = read_json(index_manifest_path)
    integrity = index_manifest.get("integrity", {})
    if not all(
        (
            integrity.get("bm25_document_count_matches"),
            integrity.get("vector_map_count_matches"),
            integrity.get("ngram_shards_verified"),
            integrity.get("chunk_id_duplicates") == 0,
            integrity.get("invalid_text_hashes") == 0,
        )
    ):
        raise RuntimeError("Full index integrity manifest is not a PASS")
    DEPLOY_ROOT.mkdir(parents=True, exist_ok=True)
    materialization: dict[str, str] = {}
    sources = [
        FULL_ARTIFACT / "metadata.sqlite3",
        FULL_ARTIFACT / "sparse_build_status.json",
        FULL_ARTIFACT / "VERSION",
    ]
    if index_manifest["faiss"]["built"]:
        sources.extend(
            [
                FULL_ARTIFACT / "faiss" / "index.faiss",
                FULL_ARTIFACT / "faiss" / "vector_ids.json",
            ]
        )
    sources.extend(sorted((FULL_ARTIFACT / "ngram").rglob("*")))
    sources.extend(sorted((FULL_ARTIFACT / "manifests").rglob("*")))
    for source in sources:
        if not source.is_file():
            continue
        relative = source.relative_to(FULL_ARTIFACT)
        destination = DEPLOY_ROOT / relative
        method = hardlink_or_copy(source, destination)
        materialization[relative.as_posix()] = method
    if index_manifest["faiss"]["built"]:
        model_manifest = read_json(
            FULL_ARTIFACT / "faiss" / "vector_manifest.json"
        )
        model_path = Path(model_manifest["model_local_path"])
        for source in sorted(model_path.rglob("*")):
            if not source.is_file() or ".cache" in source.parts:
                continue
            relative = Path("model") / source.relative_to(model_path)
            method = hardlink_or_copy(source, DEPLOY_ROOT / relative)
            materialization[relative.as_posix()] = method
        deploy_vector_manifest = {
            **model_manifest,
            "model_local_path": "../model",
        }
        write_json(
            DEPLOY_ROOT / "faiss" / "vector_manifest.json",
            deploy_vector_manifest,
        )
        materialization["faiss/vector_manifest.json"] = "generated-portable-config"
    requirements = [
        "fastapi",
        "uvicorn",
        "numpy",
        "python-Levenshtein",
        "zhconv",
    ]
    if index_manifest["faiss"]["built"]:
        requirements.extend(
            ["faiss-cpu==1.14.3", "sentence-transformers==5.6.0"]
        )
    (DEPLOY_ROOT / "requirements.txt").write_text(
        "\n".join(requirements + [""]),
        encoding="utf-8",
    )
    config = {
        "rag_version": "rag-v2-full-1.0.0",
        "database": "metadata.sqlite3",
        "ngram_root": "ngram",
        "vector_dir": "faiss" if index_manifest["faiss"]["built"] else None,
        "model_dir": "model" if index_manifest["faiss"]["built"] else None,
        "vector_enabled": False,
        "vector_decision": (
            "VECTOR_INDEX_BUILT_BUT_DISABLED_NO_GAIN"
            if index_manifest["faiss"]["built"]
            else "VECTOR_DATABASE_NOT_REQUIRED"
        ),
        "source_version": "CBETA Normal Text 2025 R3",
        "source_commit": "2ffcfdb5011e97637295c6589061b9779273ddda",
    }
    write_json(DEPLOY_ROOT / "config.json", config)
    (DEPLOY_ROOT / "VERSION").write_text("rag-v2-full-1.0.0\n", encoding="utf-8")
    (DEPLOY_ROOT / "start.ps1").write_text(
        "param([string]$ProjectRoot=(Resolve-Path "
        "(Join-Path $PSScriptRoot '..\\..')).Path)\n"
        "$ErrorActionPreference='Stop'\n"
        "$env:RAG_V2_DEPLOY_ROOT=$PSScriptRoot\n"
        "$env:RAG_V2_PROJECT_ROOT=$ProjectRoot\n"
        "$env:RAG_V2_VECTOR_ENABLED='false'\n"
        "python -m uvicorn --app-dir $ProjectRoot demo.backend.main:app "
        "--host 127.0.0.1 --port 8000\n",
        encoding="utf-8",
    )
    (DEPLOY_ROOT / "healthcheck.py").write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        "project_root = Path(os.environ.get("
        "'RAG_V2_PROJECT_ROOT', root.parents[1])).resolve()\n"
        "sys.path.insert(0, str(project_root))\n"
        "from retrieval.rag_v2.full_retriever import FullRagV2Retriever\n"
        "rag = FullRagV2Retriever(root/'metadata.sqlite3', root/'ngram', "
        "root/'faiss', vector_enabled=False)\n"
        "health = rag.health()\n"
        "assert health['loaded'] and health['ngram_ready'] and health['bm25_ready']\n"
        "print(health)\n",
        encoding="utf-8",
    )
    (DEPLOY_ROOT / "README.md").write_text(
        "# CBETA RAG V2 deployment assets\n\n"
        "This directory contains the verified full-corpus runtime indexes. "
        "Keep it at `artifacts/rag_v2_deploy` inside a checkout of the matching "
        "project commit, or set `RAG_V2_PROJECT_ROOT` to that checkout. Install "
        "`requirements.txt`, then run `python healthcheck.py` before "
        "`start.ps1`.\n\n"
        + (
            "FAISS is packaged but disabled by default because the fixed "
            "ablation showed no measurable gain over sparse hybrid retrieval.\n"
            if index_manifest["faiss"]["built"]
            else "Full-corpus FAISS is not packaged: the fixed T0278 ablation "
            "showed no gain and the full sparse evaluation passed.\n"
        ),
        encoding="utf-8",
    )
    for relative in (
        "requirements.txt",
        "config.json",
        "VERSION",
        "start.ps1",
        "healthcheck.py",
        "README.md",
    ):
        materialization[relative] = "generated"
    manifest_name = "rag_v2_deploy_manifest.json"
    assets = []
    for path in sorted(DEPLOY_ROOT.rglob("*")):
        if not path.is_file() or path.name == manifest_name:
            continue
        relative = path.relative_to(DEPLOY_ROOT).as_posix()
        assets.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "materialization": materialization.get(relative, "existing"),
                "runtime_required": relative not in {"README.md"},
            }
        )
    manifest = {
        "version": "rag-v2-full-1.0.0",
        "built_at": now_iso(),
        "source_version": index_manifest["source_version"],
        "source_commit": index_manifest["source_commit"],
        "build_commit": index_manifest["build_commit"],
        "assets": assets,
        "logical_bytes": sum(item["bytes"] for item in assets),
        "hardlink_count": sum(item["materialization"] == "hardlink" for item in assets),
        "copy_count": sum(item["materialization"] == "copy" for item in assets),
        "disk_after": disk_usage(),
    }
    write_json(DEPLOY_ROOT / manifest_name, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
