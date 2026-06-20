import argparse
from pathlib import Path

import torch
from PIL import Image

from xray_ai.data import image_transform
from xray_ai.model import build_model


def load_predictor(checkpoint: str | Path, device: str | None = None):
    """Reconstruye arquitectura, etiquetas y pesos como una sola unidad reproducible."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    labels = saved["labels"]
    model = build_model(len(labels), pretrained=False, freeze_backbone=False)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), labels, device


@torch.inference_mode()
def predict(image: Image.Image, model, labels: list[str], device: str) -> dict[str, float]:
    """Convierte logits en probabilidades; inferencia desactiva gradientes y ahorra memoria."""
    tensor = image_transform()(image).unsqueeze(0).to(device)
    probabilities = model(tensor).softmax(1).squeeze(0).cpu().tolist()
    return dict(zip(labels, probabilities, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice una imagen con un checkpoint entrenado")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="models/best.pt")
    args = parser.parse_args()
    model, labels, device = load_predictor(args.checkpoint)
    scores = predict(Image.open(args.image), model, labels, device)
    for label, probability in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{label}: {probability:.2%}")


if __name__ == "__main__":
    main()

