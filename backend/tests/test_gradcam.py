import numpy as np
import torch
from PIL import Image

from app.models.gradcam import GradCAM, generate_gradcam_overlay
from app.models.ml_model import build_resnet18


def test_gradcam_returns_pil_image():
    model = build_resnet18(num_classes=2, pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)

    with GradCAM(model) as cam:
        overlay = cam.generate(tensor, class_index=1)

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"
    assert overlay.size == (224, 224)


def test_generate_gradcam_overlay_helper():
    model = build_resnet18(num_classes=2, pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)
    overlay = generate_gradcam_overlay(model, tensor)
    assert isinstance(overlay, Image.Image)


def test_describe_region_heuristic():
    from app.services.gradcam_service import GradCAMService

    # Warm red blob in lower-right → lower-right description
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[80:95, 80:95, 0] = 255
    desc = GradCAMService._describe_from_overlay(Image.fromarray(arr))
    assert "lower" in desc
    assert "right" in desc
