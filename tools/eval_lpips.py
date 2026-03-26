import os
import numpy as np
from PIL import Image
import torch
import lpips
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img

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

def compute_average_metrics(output_dir, lpips_model):
    lpips_values = []

    for img_name in os.listdir(output_dir):
        if not img_name.endswith('_adv_image.png'):
            continue

        base_name = img_name.replace('_adv_image.png', '_originImage.png')
        adv_img_path = os.path.join(output_dir, img_name)
        original_img_path = os.path.join(output_dir, base_name)

        if not os.path.exists(original_img_path):
            continue

        adv_img = preprocess_image(adv_img_path)
        original_img = preprocess_image(original_img_path)

        lpips_value = compute_lpips(adv_img, original_img, lpips_model)
        lpips_values.append(lpips_value)

    return np.mean(lpips_values) if lpips_values else 0

if __name__ == "__main__":
    output_dir = "output/resnet_qwen3_vl/15"
    lpips_model = lpips.LPIPS(net='alex').cuda()
    lpips_val = compute_average_metrics(output_dir, lpips_model)
    print(f"{output_dir} LPIPS: {lpips_val:.4f}")
