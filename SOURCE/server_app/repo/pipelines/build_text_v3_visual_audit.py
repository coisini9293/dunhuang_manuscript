#!/usr/bin/env python3
"""Create full-page and 100%-scale slot audit artifacts for Hybrid Text V3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--missing-mask", required=True)
    ap.add_argument("--protected-mask", required=True)
    ap.add_argument("--slot-manifest", required=True)
    ap.add_argument("--glyph-provenance", required=True)
    ap.add_argument("--label-font", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paper_path, final_path = Path(args.paper), Path(args.final)
    paper = Image.open(paper_path).convert("RGB")
    final = Image.open(final_path).convert("RGB")
    missing = np.asarray(Image.open(args.missing_mask).convert("L")) > 127
    protected = np.asarray(Image.open(args.protected_mask).convert("L")) > 127
    pa, fa = np.asarray(paper), np.asarray(final)
    changed = np.any(pa != fa, axis=2)
    manifest = json.loads(Path(args.slot_manifest).read_text(encoding="utf-8"))
    provenance = json.loads(Path(args.glyph_provenance).read_text(encoding="utf-8"))
    prov = {x["slot_id"]: x for x in provenance["slots"]}

    # Exact-resolution side-by-side full page.
    full = Image.new("RGB", (paper.width * 2 + 8, paper.height), "black")
    full.paste(paper, (0, 0)); full.paste(final, (paper.width + 8, 0))
    full.save(out / "comparison_full.png")

    overlay = np.asarray(final).copy()
    red = np.zeros_like(overlay); red[..., 0] = 255
    overlay[missing] = (overlay[missing] * .62 + red[missing] * .38).astype(np.uint8)
    Image.fromarray(overlay).save(out / "missing_overlay.png")
    pdiff = np.zeros_like(pa)
    pdiff[changed & protected] = [255, 0, 255]
    Image.fromarray(pdiff).save(out / "protected_diff.png")

    slots = manifest["slots"]
    label_font = ImageFont.truetype(args.label_font, 17)
    columns = 4
    tile_w, tile_h = 330, 205
    rows = (len(slots) + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile_w, rows * tile_h), (28, 28, 28))
    draw = ImageDraw.Draw(montage)
    slot_records = []
    for i, slot in enumerate(slots):
        x1, y1, x2, y2 = slot["bbox"]
        margin = 7
        box = (max(0, x1 - margin), max(0, y1 - margin), min(paper.width, x2 + margin), min(paper.height, y2 + margin))
        before = paper.crop(box); after = final.crop(box)
        scale = min(2.0, 140 / max(before.width, before.height))
        size = (max(1, int(before.width * scale)), max(1, int(before.height * scale)))
        before = before.resize(size, Image.Resampling.NEAREST)
        after = after.resize(size, Image.Resampling.NEAREST)
        col, row = i % columns, i // columns
        ox, oy = col * tile_w, row * tile_h
        draw.rectangle((ox, oy, ox + tile_w - 2, oy + tile_h - 2), outline=(80, 80, 80), width=1)
        changed_pixels = int(changed[y1:y2, x1:x2].sum())
        method = prov[slot["slot_id"]]["method"]
        status = "NON_TEXT" if not slot["render_required"] else ("EDIT" if changed_pixels else "NO_LEGAL_STROKE_INTERSECTION")
        label = f"{slot['slot_id']}  {slot['selected_text'] or '∅'}  {status}"
        draw.text((ox + 6, oy + 5), label, fill=(238, 238, 238), font=label_font)
        draw.text((ox + 6, oy + 28), "BEFORE", fill=(180, 180, 180), font=label_font)
        draw.text((ox + 170, oy + 28), "AFTER", fill=(180, 180, 180), font=label_font)
        montage.paste(before, (ox + 6, oy + 52))
        montage.paste(after, (ox + 170, oy + 52))
        slot_records.append({
            "slot_id": slot["slot_id"], "selected_text": slot["selected_text"],
            "bbox": slot["bbox"], "render_required": slot["render_required"],
            "changed_pixels": changed_pixels, "visual_status": status, "glyph_method": method,
        })
    montage.save(out / "comparison_slots.png")

    audit = {
        "status": "READY_FOR_VISUAL_REVIEW",
        "case_id": manifest["case_id"],
        "dimensions_unchanged": paper.size == final.size,
        "protected_changed_pixels": int((changed & protected).sum()),
        "outside_missing_changed_pixels": int((changed & ~missing).sum()),
        "inside_missing_changed_pixels": int((changed & missing).sum()),
        "text_slots": sum(x["render_required"] for x in slots),
        "non_text_slots": sum(not x["render_required"] for x in slots),
        "no_legal_stroke_intersection_slots": sum(x["visual_status"] == "NO_LEGAL_STROKE_INTERSECTION" for x in slot_records),
        "paper_sha256": sha256(paper_path), "final_sha256": sha256(final_path),
        "slot_records": slot_records,
    }
    (out / "visual_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "visual_audit.md").write_text(
        "# Hybrid Text V3 Visual Audit\n\n"
        f"- Case: `{audit['case_id']}`\n"
        f"- Status: `{audit['status']}`\n"
        f"- Protected changed pixels: `{audit['protected_changed_pixels']}`\n"
        f"- Outside missing-mask changed pixels: `{audit['outside_missing_changed_pixels']}`\n"
        f"- Inside missing-mask changed pixels: `{audit['inside_missing_changed_pixels']}`\n"
        f"- No-legal-stroke-intersection slots: `{audit['no_legal_stroke_intersection_slots']}`\n\n"
        "`NO_LEGAL_STROKE_INTERSECTION` means the frozen missing pixels do not overlap the approved glyph realization; no out-of-mask ink was invented.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: audit[k] for k in audit if k != "slot_records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
