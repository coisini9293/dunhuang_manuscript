from collections import deque
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "面向残片缀合的敦煌文献图像数据集" / "人工数据组"
OUTPUT = ROOT / "preview-sandbox" / "assets" / "about-fragments"

# One deliberately varied source per manuscript; avoid sequential variants of the same sheet.
SOURCES = {
    "dy054-2": ("Dy054(2)", "Dy054(2)v1_1.png"),
    "dy043": ("Dy043", "Dy043v1_2.png"),
    "dy020-1": ("Dy020(1)", "Dy020(1)v1_3.png"),
    "dy011": ("Dy011", "Dy011v1_1.png"),
    "dy044": ("Dy044", "Dy044v1_3.png"),
    "dy019": ("Dy019", "Dy019v1_3.png"),
    "dy056-3": ("Dy056(3)", "Dy056(3)v1_4.png"),
    "dy017": ("Dy017", "Dy017v1_2.png"),
    "dy042": ("Dy042", "Dy042v1_4.png"),
    "dy016": ("Dy016", "Dy016v1_3.png"),
    "dy002": ("Dy002", "Dy002v1_2.png"),
    "dy060": ("Dy060", "Dy060v1_3.png"),
    "dy058": ("Dy058", "Dy058v1_4.png"),
    "dy051": ("Dy051", "Dy051v1_2.png"),
}


def is_background(pixel):
    r, g, b, a = pixel
    if a < 18:
        return True
    spread = max(r, g, b) - min(r, g, b)
    return (max(r, g, b) < 42 and spread < 18) or (min(r, g, b) > 244 and spread < 15)


def remove_edge_background(image):
    image = image.convert("RGBA")
    width, height = image.size
    pixels = image.load()
    seen = bytearray(width * height)
    queue = deque()

    def add(x, y):
        index = y * width + x
        if not seen[index] and is_background(pixels[x, y]):
            seen[index] = 1
            queue.append((x, y))

    for x in range(width):
        add(x, 0)
        add(x, height - 1)
    for y in range(1, height - 1):
        add(0, y)
        add(width - 1, y)

    while queue:
        x, y = queue.popleft()
        pixels[x, y] = (0, 0, 0, 0)
        if x:
            add(x - 1, y)
        if x + 1 < width:
            add(x + 1, y)
        if y:
            add(x, y - 1)
        if y + 1 < height:
            add(x, y + 1)

    # Dataset scans occasionally include isolated measurement rulers.  Keep the
    # dominant connected manuscript body so those detached border artifacts do
    # not become visible as lines beside a fragment.
    foreground = bytearray(width * height)
    components = []
    for start_y in range(height):
        for start_x in range(width):
            start = start_y * width + start_x
            if foreground[start] or pixels[start_x, start_y][3] < 40:
                continue
            foreground[start] = 1
            component = [start]
            component_queue = deque([(start_x, start_y)])
            while component_queue:
                x, y = component_queue.popleft()
                for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    index = next_y * width + next_x
                    if foreground[index] or pixels[next_x, next_y][3] < 40:
                        continue
                    foreground[index] = 1
                    component.append(index)
                    component_queue.append((next_x, next_y))
            components.append(component)

    if components:
        largest = max(components, key=len)
        keep = bytearray(width * height)
        for index in largest:
            keep[index] = 1
        for y in range(height):
            for x in range(width):
                if not keep[y * width + x]:
                    pixels[x, y] = (0, 0, 0, 0)

    alpha = image.getchannel("A")
    bounds = alpha.getbbox()
    if not bounds:
        raise ValueError("background removal left no visible fragment")
    left, top, right, bottom = bounds
    image = image.crop((max(0, left - 2), max(0, top - 2), min(width, right + 2), min(height, bottom + 2)))
    image.thumbnail((640, 420), Image.Resampling.LANCZOS)
    return image


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (folder, filename) in SOURCES.items():
        source = DATASET / folder / filename
        output = OUTPUT / f"{name}.png"
        remove_edge_background(Image.open(source)).save(output, "PNG", optimize=True)
        print(f"{source.relative_to(ROOT)} -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
