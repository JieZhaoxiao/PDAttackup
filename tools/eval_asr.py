import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from other_attacks import model_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output/resnet_qwen3_vl/15", type=str)
    parser.add_argument("--label_file", default="imagenet_compatible/labels.txt", type=str)
    parser.add_argument(
        "--dataset_name",
        default="imagenet_compatible",
        choices=["imagenet_compatible", "cub_200_2011", "standford_car"],
        type=str,
    )
    parser.add_argument("--res", default=224, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    return parser.parse_args()


def get_eval_models(dataset_name):
    if dataset_name == "imagenet_compatible":
        return [
            "resnet",
            "vgg",
            "mobile",
            "inception",
            "convnext",
            "vit",
            "swin",
            "deit-b",
            "deit-s",
            "mixer-b",
            "mixer-l",
            "tf2torch_adv_inception_v3",
            "tf2torch_ens3_adv_inc_v3",
            "tf2torch_ens4_adv_inc_v3",
            "tf2torch_ens_adv_inc_res_v2",
        ], 1000
    if dataset_name == "cub_200_2011":
        return ["cubResnet50", "cubSEResnet154", "cubSEResnet101"], 200
    if dataset_name == "standford_car":
        return ["carResnet50", "carSEResnet154", "carSEResnet101"], 196
    raise NotImplementedError


def load_labels(label_file):
    with open(label_file, "r", encoding="utf-8") as file:
        return np.array([int(line.strip()) - 1 for line in file.readlines() if line.strip()])


def load_adv_images(output_dir, res):
    image_paths = sorted(
        [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.endswith("_adv_image.png")
        ]
    )
    images = []
    for path in image_paths:
        image = torch.tensor(0)
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image = image.resize((res, res), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        images.append(image)

    if not images:
        raise FileNotFoundError(f"No *_adv_image.png files found in {output_dir}")

    return np.concatenate(images, axis=0)


def evaluate_attack(adv_img, label, res, output_dir, dataset_name, batch_size):
    model_names, nb_classes = get_eval_models(dataset_name)
    log_path = os.path.join(output_dir, "eval_asr_log.txt")
    log = open(log_path, mode="w", encoding="utf-8")

    for name in model_names:
        print(f"\n*********Transfer to {name}********")
        print(f"\n*********Transfer to {name}********", file=log)
        model = model_selection(name)
        model.eval()
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=nb_classes,
            preprocessing=(
                np.array([0.5, 0.5, 0.5]),
                np.array([0.5, 0.5, 0.5]),
            )
            if "adv" in name
            else (
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]),
            ),
            device_type="gpu",
        )

        adv_pred = classifier.predict(adv_img, batch_size=batch_size)
        accuracy = (
            np.sum((np.argmax(adv_pred, axis=1) - 1) == label) / len(label)
            if "adv" in name
            else np.sum(np.argmax(adv_pred, axis=1) == label) / len(label)
        )
        asr = 100 - accuracy * 100
        print(f"Attack Success Rate: {asr}%")
        print(f"Attack Success Rate: {asr}%", file=log)

    log.close()


def main():
    args = parse_args()
    label = load_labels(args.label_file)
    adv_img = load_adv_images(args.output_dir, args.res)

    if len(adv_img) != len(label):
        raise ValueError(
            f"Number of adversarial images ({len(adv_img)}) does not match labels ({len(label)})."
        )

    evaluate_attack(
        adv_img=adv_img,
        label=label,
        res=args.res,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
