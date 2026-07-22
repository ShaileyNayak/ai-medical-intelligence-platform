import numpy as np
import torch
from PIL import Image

from app.models.chest_xray.gradcam import GradCAM, generate_gradcam_overlay
from app.models._shared import build_resnet18
from app.models.registry import get_model, reset_registry


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

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[80:95, 80:95, 0] = 255
    desc = GradCAMService._describe_from_overlay(Image.fromarray(arr), scan_type="chest_xray")
    assert "lower" in desc
    assert "right" in desc


def test_registry_get_model_by_scan_type():
    reset_registry()
    chest = get_model("chest_xray")
    brain = get_model("brain_mri")
    skin = get_model("skin_lesion")
    assert chest.scan_type == "chest_xray"
    assert brain.scan_type == "brain_mri"
    assert skin.scan_type == "skin_lesion"
    assert get_model("chest_xray") is chest
