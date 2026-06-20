import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from xray_ai.data import balanced_class_weights, make_loaders
from xray_ai.metrics import classification_metrics, confusion_matrix
from xray_ai.model import build_model


def seed_everything(seed: int = 42) -> None:
    """Fija las fuentes principales de azar para poder comparar experimentos."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Una sola función sirve para entrenar (con optimizer) o validar (sin él)."""
    training = optimizer is not None
    model.train(training)
    loss_sum, truths, guesses = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)  # Menos memoria que escribir ceros.
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        truths.append(labels.detach().cpu())
        guesses.append(logits.argmax(1).detach().cpu())

    truth, guess = torch.cat(truths), torch.cat(guesses)
    metrics = classification_metrics(confusion_matrix(truth, guess, len(loader.dataset.classes)))
    return loss_sum / len(loader.dataset), metrics


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, labels = make_loaders(args.data, args.batch_size, args.workers)
    model = build_model(
        len(labels), pretrained=not args.no_pretrained, freeze_backbone=not args.unfreeze
    )
    model.to(device)

    weights = balanced_class_weights(train_loader.dataset.targets, len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    output, best_loss = Path(args.output), float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, device)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                          "train": train_metrics, "val": val_metrics}, ensure_ascii=False))
        if val_loss < best_loss:  # Guardamos el mejor, no simplemente el último.
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "labels": labels,
                        "image_size": 224, "val_metrics": val_metrics}, output)

    print(f"Mejor modelo guardado en {output} | dispositivo: {device}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador educativo de radiografías"
    )
    parser.add_argument("--data", default="data/demo")
    parser.add_argument("--output", default="models/best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze", action="store_true", help="Ajusta también la red base")
    parser.add_argument("--no-pretrained", action="store_true", help="No descarga pesos ImageNet")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
