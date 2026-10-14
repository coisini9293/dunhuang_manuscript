"""Pick a matplotlib font that supports Chinese and Latin text."""
from __future__ import annotations

import matplotlib
from matplotlib import font_manager

# macOS / Windows / Linux common CJK fonts, in priority order.
_FONT_CANDIDATES = (
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "STHeiti",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
)

_configured = False


def configure_matplotlib_font() -> str:
    """Configure sans-serif stack for CJK labels. Returns the chosen font name."""
    global _configured
    if _configured:
        return matplotlib.rcParams["font.sans-serif"][0]

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in _FONT_CANDIDATES if name in available), "DejaVu Sans")

    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _configured = True
    return chosen
