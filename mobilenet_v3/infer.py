"""MobileNetV3 图像分类推理（torchvision 预训练权重）。

用法:
    python -m mobilenet_v3.infer                      # small + 示例图
    python -m mobilenet_v3.infer --variant large --src cat.jpg
"""

import argparse

import torch
from PIL import Image
from torchvision import models


def load_model(variant: str):
    if variant == "small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
    model.eval()
    return model, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None)
    parser.add_argument("--variant", choices=["small", "large"], default="small")
    args = parser.parse_args()

    from common.utils import download, get_device

    src = args.src or download("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    device = get_device()

    model, weights = load_model(args.variant)
    model = model.to(device)

    preprocess = weights.transforms()
    img = Image.open(src).convert("RGB")
    batch = preprocess(img)[None].to(device)

    with torch.no_grad():
        logits = model(batch)
    probs = logits.softmax(1)[0]
    top5 = torch.topk(probs, 5)
    print(f"[RESULT] {src} ({args.variant})")
    for p, i in zip(top5.values, top5.indices):
        print(f"  {weights.meta['categories'][i]:<30s} {p.item():.2%}")


if __name__ == "__main__":
    main()
