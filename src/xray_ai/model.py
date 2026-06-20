from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Adapta EfficientNet-B0 a N clases usando transferencia de aprendizaje."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.features.requires_grad_(not freeze_backbone)  # Congelar ahorra tiempo y datos.
    inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(inputs, classes)
    return model

