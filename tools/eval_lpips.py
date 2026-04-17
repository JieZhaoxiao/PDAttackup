import argparse
import os
import warnings

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output/resnet_qwen3_vl/15", type=str)
    parser.add_argument("--res", default=224, type=int)
    return parser.parse_args()


def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def compute_lpips(image1, image2, model):
    transform = transforms.Compose([transforms.ToTensor()])
    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)
    device = next(model.parameters()).device
    image1 = image1.to(device)
    image2 = image2.to(device)
    with torch.no_grad():
        lpips_value = model(image1, image2)
    return lpips_value.item()


def compute_average_metrics(output_dir, lpips_model, res):
    lpips_values = []

    for img_name in os.listdir(output_dir):
        if not img_name.endswith("_adv_image.png"):
            continue

        base_name = img_name.replace("_adv_image.png", "_originImage.png")
        adv_img_path = os.path.join(output_dir, img_name)
        original_img_path = os.path.join(output_dir, base_name)

        if not os.path.exists(original_img_path):
            continue

        adv_img = preprocess_image(adv_img_path, target_size=(res, res))
        original_img = preprocess_image(original_img_path, target_size=(res, res))
        lpips_value = compute_lpips(adv_img, original_img, lpips_model)
        lpips_values.append(lpips_value)

    return np.mean(lpips_values) if lpips_values else 0


def main():
    args = parse_args()
    lpips_model = lpips.LPIPS(net="alex").cuda()
    lpips_val = compute_average_metrics(args.output_dir, lpips_model, args.res)
    print(f"{args.output_dir} LPIPS: {lpips_val:.4f}")


if __name__ == "__main__":
    main()
