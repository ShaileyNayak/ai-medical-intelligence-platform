import numpy as np
import torch
from PIL import Image

from app.models._shared import build_resnet18
from app.models.brain_mri.gradcam import GradCAM as BrainGradCAM
from app.models.brain_mri.gradcam import get_target_layer as brain_target_layer
from app.models.chest_xray.gradcam import GradCAM as ChestGradCAM
from app.models.chest_xray.gradcam import generate_gradcam_overlay
from app.models.chest_xray.gradcam import get_target_layer as chest_target_layer
from app.models.registry import get_model, reset_registry
from app.utils.image_processing import blend_heatmap_overlay


def test_chest_gradcam_returns_pil_image():
    model = build_resnet18(num_classes=4, pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)

    with ChestGradCAM(model, multi_label=True) as cam:
        overlay = cam.generate(tensor)  # highest sigmoid prob class

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"
    assert overlay.size == (224, 224)


def test_chest_gradcam_uses_top_probability_class():
    model = build_resnet18(num_classes=4, pretrained=False)
    model.eval()
    # Bias fc so class 2 has the highest logit → highest sigmoid prob
    with torch.no_grad():
        model.fc.bias.zero_()
        model.fc.bias[2] = 10.0

    tensor = torch.randn(1, 3, 224, 224)
    cam = ChestGradCAM(model, multi_label=True)
    selected = cam._select_class_index(model(tensor), class_index=None)
    cam.close()
    assert selected == 2


def test_brain_gradcam_returns_pil_image():
    model = build_resnet18(num_classes=2, pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)

    with BrainGradCAM(model) as cam:
        overlay = cam.generate(tensor, class_index=0)

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"


def test_target_layers_are_layer4_final_block():
    model = build_resnet18(num_classes=2, pretrained=False)
    assert chest_target_layer(model) is model.layer4[-1]
    assert brain_target_layer(model) is model.layer4[-1]


def test_blend_heatmap_overlay_shared_util():
    cam = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    base = Image.new("RGB", (64, 64), color=(40, 40, 40))
    overlay = blend_heatmap_overlay(cam, base, alpha=0.5)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (64, 64)


def test_generate_gradcam_overlay_helper():
    model = build_resnet18(num_classes=4, pretrained=False)
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


def test_lazy_load_status_does_not_instantiate_modules():
    from app.models.registry import _instances, models_loaded_status

    reset_registry()
    status = models_loaded_status()
    assert status == {
        "chest_xray": False,
        "brain_mri": False,
        "skin_lesion": False,
    }
    assert _instances == {}

    chest = get_model("chest_xray")
    assert "chest_xray" in _instances
    assert "brain_mri" not in _instances
    assert "skin_lesion" not in _instances
    assert get_model("chest_xray") is chest
    assert models_loaded_status()["brain_mri"] is False
