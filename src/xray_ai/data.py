from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def image_transform(training: bool = False) -> transforms.Compose:
    """Convierte cualquier radiografía a la entrada RGB esperada por EfficientNet."""
    augment = [transforms.RandomRotation(3)] if training else []
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *augment,  # Aumentación solo al entrenar; validar debe ser determinista.
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def make_loaders(root: str | Path, batch_size: int = 16, workers: int = 0):
    """Lee data/{train,val}/{clase} y devuelve lotes eficientes y sus etiquetas."""
    root = Path(root)
    train = datasets.ImageFolder(root / "train", image_transform(training=True))
    val = datasets.ImageFolder(root / "val", image_transform())
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("train y val deben contener exactamente las mismas carpetas de clase")

    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train, shuffle=True, **options),
        DataLoader(val, shuffle=False, **options),
        train.classes,
    )


def balanced_class_weights(targets: list[int], classes: int) -> torch.Tensor:
    """Da más peso a clases escasas para que la mayoría no domine el aprendizaje."""
    counts = torch.bincount(torch.tensor(targets), minlength=classes).float()
    if (counts == 0).any():
        raise ValueError("Cada clase necesita al menos una imagen de entrenamiento")
    return counts.sum() / (classes * counts)

