import torch
from PIL import Image

from xray_ai.data import IMAGE_SIZE, balanced_class_weights, image_transform
from xray_ai.metrics import classification_metrics, confusion_matrix
from xray_ai.model import build_model


def test_transform_has_expected_shape():
    tensor = image_transform()(Image.new("L", (100, 80)))
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_balanced_weights_favor_minority():
    weights = balanced_class_weights([0, 0, 0, 1], classes=2)
    assert weights[1] > weights[0]


def test_metrics_are_perfect_for_perfect_predictions():
    truth = predicted = torch.tensor([0, 1, 1, 0])
    metrics = classification_metrics(confusion_matrix(truth, predicted, classes=2))
    assert metrics["accuracy"] == metrics["macro_f1"] == 1.0


def test_model_output_shape():
    model = build_model(classes=2, pretrained=False)
    assert model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape == (1, 2)

