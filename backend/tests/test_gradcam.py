import numpy as np

from app.services.gradcam_service import GradCAMService


def test_describe_region():
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[80, 80] = 1.0
    desc = GradCAMService._describe_region(cam)
    assert "lower" in desc
    assert "right" in desc
