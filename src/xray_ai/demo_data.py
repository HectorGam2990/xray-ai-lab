import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_image(kind: str, rng: np.random.Generator, size: int = 224) -> Image.Image:
    """Crea patrones falsos para probar el pipeline sin usar datos de pacientes."""
    pixels = rng.normal(35, 10, (size, size)).clip(0, 255).astype("uint8")
    image = Image.fromarray(pixels, mode="L")
    draw = ImageDraw.Draw(image)
    x, y, radius = rng.integers(70, 110), rng.integers(70, 110), rng.integers(35, 55)
    box = (x - radius, y - radius, x + radius, y + radius)
    fill = int(rng.integers(150, 220))
    draw.ellipse(box, fill=fill) if kind == "CIRCLE" else draw.rectangle(box, fill=fill)
    return image


def generate(root: str | Path, train_count: int = 24, val_count: int = 8, seed: int = 42) -> None:
    root, rng = Path(root), np.random.default_rng(seed)
    for split, count in {"train": train_count, "val": val_count}.items():
        for label in ("CIRCLE", "SQUARE"):
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                make_image(label, rng).save(folder / f"{index:03}.png")
    print(f"Datos sintéticos creados en {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera datos sintéticos para aprender el flujo")
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.train, args.val, args.seed)


if __name__ == "__main__":
    main()

