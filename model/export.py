"""Export model to TorchScript or ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model_def import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to TorchScript or ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(num_classes=args.num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "torchscript":
        scripted = torch.jit.trace(model, dummy)
        scripted.save(str(args.out))
    else:
        torch.onnx.export(
            model,
            dummy,
            str(args.out),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    print(f"Exported {args.format} model -> {args.out}")


if __name__ == "__main__":
    main()
