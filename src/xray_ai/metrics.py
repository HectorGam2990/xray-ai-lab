import torch


def confusion_matrix(truth: torch.Tensor, predicted: torch.Tensor, classes: int) -> torch.Tensor:
    """Filas=valor real y columnas=predicción; base de casi todas las métricas."""
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    for real, guess in zip(truth.cpu(), predicted.cpu(), strict=True):
        matrix[real, guess] += 1
    return matrix


def classification_metrics(matrix: torch.Tensor) -> dict[str, object]:
    """Resume rendimiento sin ocultar el comportamiento de cada clase."""
    matrix = matrix.float()
    diagonal = matrix.diag()
    recall = diagonal / matrix.sum(1).clamp_min(1)
    precision = diagonal / matrix.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": (diagonal.sum() / matrix.sum().clamp_min(1)).item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "recall_per_class": recall.tolist(),
        "precision_per_class": precision.tolist(),
    }

