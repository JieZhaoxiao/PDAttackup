import argparse
import glob
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline
from natsort import ns, natsorted

import aadattack
from attention import AttentionControlEdit
from other_attacks import model_transfer

start_time = time.time()
parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_dir",
    default="output/resnet_qwen3_vl/15",
    type=str,
    help="Where to save the adversarial examples, and other results",
)
parser.add_argument(
    "--context_prompt_path",
    default="Text/qwen3_vl/15_tokens.txt",
    type=str,
    help="Path to the 15-token context prompt file",
)
parser.add_argument(
    "--foreground_prompt_path",
    default="Text/qwen3_vl/15_tokens_fg.txt",
    type=str,
    help="Path to the foreground prompt file",
)
parser.add_argument(
    "--background_prompt_path",
    default="Text/qwen3_vl/15_tokens_bg.txt",
    type=str,
    help="Path to the background prompt file",
)
parser.add_argument(
    "--images_root",
    default="imagenet_compatible/images",
    type=str,
    help="The clean images root directory",
)
parser.add_argument(
    "--label_path",
    default="imagenet_compatible/labels.txt",
    type=str,
    help="The clean images labels.txt",
)
parser.add_argument(
    "--is_test",
    default=False,
    type=bool,
    help="Whether to test the robustness of the generated adversarial examples",
)
parser.add_argument(
    "--skip_transfer_eval",
    action="store_true",
    help="Only generate adversarial examples without running transfer evaluation",
)
parser.add_argument(
    "--pretrained_diffusion_path",
    default="stabilityai/stable-diffusion-2-1-base",
    type=str,
    help="Path to the Stable Diffusion 2.1 Base checkpoint",
)
parser.add_argument(
    "--vlm_model_path",
    default="Qwen3-VL-8B-Instruct",
    type=str,
    help="Local Qwen3-VL model path for prompt generation",
)
parser.add_argument(
    "--force_regenerate_prompts",
    action="store_true",
    help="Regenerate the 15-token prompt triplets even if the files already exist",
)
parser.add_argument("--diffusion_steps", default=20, type=int, help="Total DDIM sampling steps")
parser.add_argument("--start_step", default=15, type=int, help="Which DDIM step to start the attack")
parser.add_argument("--iterations", default=30, type=int, help="Iterations of optimizing the adversarial latent")
parser.add_argument("--res", default=224, type=int, help="Input image resized resolution")
parser.add_argument("--latent_lr", default=0.01, type=float, help="Latent optimization learning rate")
parser.add_argument(
    "--model_name",
    default="resnet",
    type=str,
    help="The surrogate model from which the adversarial examples are crafted",
)
parser.add_argument(
    "--dataset_name",
    default="imagenet_compatible",
    type=str,
    choices=["imagenet_compatible", "cub_200_2011", "standford_car"],
    help="The dataset name for generating adversarial examples",
)
parser.add_argument("--guidance", default=3, type=float, help="Guidance scale of diffusion models")
parser.add_argument("--attack_loss_weight", default=10, type=float, help="Classification loss weight")
parser.add_argument("--transfer_loss_weight", default=1000, type=float, help="Background semantic injection loss weight")
parser.add_argument("--cross_attn_loss_weight", default=100, type=float, help="Complementary cross-attention alignment loss weight")
parser.add_argument("--self_attn_loss_weight", default=100, type=float, help="Middle self-attention alignment loss weight")


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_lines(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def ensure_prompt_files(args, expected_count):
    prompt_paths = [
        args.context_prompt_path,
        args.foreground_prompt_path,
        args.background_prompt_path,
    ]

    if not args.force_regenerate_prompts and all(count_lines(path) >= expected_count for path in prompt_paths):
        return

    output_base_dir = os.path.dirname(args.context_prompt_path) or "."
    os.makedirs(output_base_dir, exist_ok=True)

    command = [
        sys.executable,
        "prompt_qwen.py",
        "--model_path",
        args.vlm_model_path,
        "--dataset_name",
        args.dataset_name,
        "--image_dir",
        args.images_root,
        "--label_file",
        args.label_path,
        "--output_base_dir",
        output_base_dir,
        "--context_filename",
        os.path.basename(args.context_prompt_path),
        "--foreground_filename",
        os.path.basename(args.foreground_prompt_path),
        "--background_filename",
        os.path.basename(args.background_prompt_path),
    ]

    print(f"[Prompt] Generating 15-token prompt triplets into {output_base_dir}...")
    subprocess.run(command, check=True)


seed_torch(42)


def run_diffusion_attack(
    image,
    label,
    diffusion_model,
    diffusion_steps,
    guidance=2.5,
    self_replace_steps=1.0,
    save_dir=r"C:\Users\PC\Desktop\output",
    res=224,
    model_name="inception",
    start_step=15,
    iterations=30,
    img_index=0,
    args=None,
):
    controller = AttentionControlEdit(diffusion_steps, self_replace_steps, args.res)

    adv_image, clean_acc, adv_acc = aadattack.aadattack(
        diffusion_model,
        label,
        controller,
        num_inference_steps=diffusion_steps,
        guidance_scale=guidance,
        image=image,
        save_path=save_dir,
        res=res,
        model_name=model_name,
        start_step=start_step,
        iterations=iterations,
        img_index=img_index,
        args=args,
    )

    return adv_image, clean_acc, adv_acc


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.res % 32 == 0 and args.res >= 96, "Please ensure the input resolution is a multiple of 32 and >= 96."

    guidance = args.guidance
    diffusion_steps = args.diffusion_steps
    start_step = args.start_step
    iterations = args.iterations
    res = args.res
    model_name = args.model_name

    if args.dataset_name == "imagenet_compatible":
        assert model_name not in [
            "cubResnet50",
            "cubSEResnet154",
            "cubSEResnet101",
            "carResnet50",
            "carSEResnet154",
            "carSEResnet101",
        ], f"There is no pretrained weight of {model_name} for ImageNet-Compatible dataset."
    if args.dataset_name == "cub_200_2011":
        assert model_name in ["cubResnet50", "cubSEResnet154", "cubSEResnet101"], (
            f"There is no pretrained weight of {model_name} for CUB_200_2011 dataset."
        )
    if args.dataset_name == "standford_car":
        assert model_name in ["carResnet50", "carSEResnet154", "carSEResnet101"], (
            f"There is no pretrained weight of {model_name} for Standford Cars dataset."
        )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(args.label_path, "r", encoding="utf-8") as file:
        label = [int(line.rstrip()) - 1 for line in file.readlines()]
        label = np.array(label)

    ensure_prompt_files(args, expected_count=len(label))

    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to("cuda:0")
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    all_images = glob.glob(os.path.join(args.images_root, "*"))
    all_images = natsorted(all_images, alg=ns.PATH)

    adv_images = []
    images = []

    if args.is_test:
        all_clean_images = natsorted(glob.glob(os.path.join(args.images_root, "*originImage*")), alg=ns.PATH)
        all_adv_images = natsorted(glob.glob(os.path.join(args.images_root, "*adv_image*")), alg=ns.PATH)
        for image_path, adv_image_path in zip(all_clean_images, all_adv_images):
            tmp_image = Image.open(image_path).convert("RGB")
            tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
            images.append(np.array(tmp_image).astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)

            tmp_image = Image.open(adv_image_path).convert("RGB")
            tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
            adv_images.append(np.array(tmp_image).astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)

        images = np.concatenate(images)
        adv_images = np.concatenate(adv_images)
        model_transfer(images, adv_images, label, res, save_path=save_dir, fid_path=args.images_root, args=args)
        sys.exit()

    for ind, image_path in enumerate(all_images):
        tmp_image = Image.open(image_path).convert("RGB")
        tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, "0") + "_originImage.png"))

        adv_image, clean_acc, adv_acc = run_diffusion_attack(
            tmp_image,
            label[ind:ind + 1],
            ldm_stable,
            diffusion_steps,
            guidance=guidance,
            res=res,
            model_name=model_name,
            start_step=start_step,
            iterations=iterations,
            save_dir=os.path.join(save_dir, str(ind).rjust(4, "0")),
            img_index=ind,
            args=args,
        )
        adv_images.append(adv_image.astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        images.append(np.array(tmp_image).astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)

    if args.skip_transfer_eval:
        sys.exit(0)

    images = np.concatenate(images)
    adv_images = np.concatenate(adv_images)
    model_transfer(images, adv_images, label, res, save_path=save_dir, args=args)

end_time = time.time()
print(f"Program finished, total elapsed time: {end_time - start_time:.2f} seconds")
