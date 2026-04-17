import argparse
import json
import os
import re

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen3-VL-8B-Instruct", type=str)
    parser.add_argument("--dataset_name", default="imagenet_compatible", choices=["imagenet_compatible", "cub_200_2011", "standford_car"])
    parser.add_argument("--image_dir", default="imagenet_compatible/images", type=str)
    parser.add_argument("--label_file", default="imagenet_compatible/labels.txt", type=str)
    parser.add_argument("--imagenet_json_path", default="imagenet_compatible/imagenet_class_index.json", type=str)
    parser.add_argument("--output_base_dir", default="Text/qwen3_vl", type=str)
    parser.add_argument("--context_filename", default="15_tokens.txt", type=str)
    parser.add_argument("--foreground_filename", default="15_tokens_fg.txt", type=str)
    parser.add_argument("--background_filename", default="15_tokens_bg.txt", type=str)
    return parser.parse_args()


def load_imagenet_classes(imagenet_json_path):
    if not os.path.exists(imagenet_json_path):
        raise FileNotFoundError(f"{imagenet_json_path}")

    with open(imagenet_json_path, "r", encoding="utf-8") as file:
        class_idx = json.load(file)

    return {int(k): v[1].replace("_", " ") for k, v in class_idx.items()}


def load_dataset_classes(args):
    if args.dataset_name == "imagenet_compatible":
        return load_imagenet_classes(args.imagenet_json_path)
    if args.dataset_name == "cub_200_2011":
        from dataset_caption import CUB_label
        return CUB_label.refined_Label
    if args.dataset_name == "standford_car":
        from dataset_caption import stanfordCar_label
        return stanfordCar_label.refined_Label
    raise NotImplementedError


def load_raw_labels(label_file):
    with open(label_file, "r", encoding="utf-8") as file:
        return [int(line.strip()) for line in file.readlines() if line.strip().isdigit()]


def load_image_files(image_dir):
    try:
        return sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
    except ValueError:
        return sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")])


def build_context_instruction(class_name):
    return (
        "Given the image and the class label "
        f"'{class_name}', write one short caption that contains the target object and exactly one non-target scene cue. "
        "Return only the caption itself, without numbering or explanation. "
        "Keep the caption under 15 tokens."
    )


def build_background_instruction(class_name):
    return (
        "Given the image and the class label "
        f"'{class_name}', output only one short non-target scene cue from the background. "
        "Do not repeat the class label. "
        "Prefer scene regions, surfaces, materials, lighting, weather, or simple context such as sky, grass, road, wall, sand, foliage, or plain background. "
        "Do not mention people, animals, vehicles, body parts, or another foreground object. "
        "Keep the answer under 8 tokens."
    )


def run_qwen_caption(model, processor, image, instruction, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return " ".join(output_text.split())


def sanitize_background_prompt(class_name, raw_caption):
    caption = " ".join(raw_caption.split())
    caption = re.compile(re.escape(class_name), re.IGNORECASE).sub("", caption)
    caption = re.sub(r"[^a-zA-Z0-9, ]+", " ", caption)
    caption = re.sub(r"\s*,\s*", ", ", caption)
    caption = " ".join(caption.split()).strip(", ")

    if not caption:
        caption = "plain background"

    banned_terms = {
        "person", "people", "man", "woman", "boy", "girl", "child", "children",
        "hand", "hands", "leg", "legs", "arm", "arms", "face", "head",
        "dog", "cat", "horse", "bird", "player", "crowd", "husky",
        "shirt", "jacket", "coat", "pants", "shoe", "shoes", "glove", "gloves", "helmet", "hat", "cap",
    }

    segments = []
    for segment in caption.split(","):
        segment = " ".join(segment.split()).strip()
        if not segment:
            continue

        words = segment.split()[:3]
        if any(word.lower() in banned_terms for word in words):
            continue

        cleaned_segment = " ".join(words).strip(", ")
        if cleaned_segment and cleaned_segment not in segments:
            segments.append(cleaned_segment)

    if not segments:
        segments = ["plain background"]

    return ", ".join(segments[:1])


def sanitize_context_prompt(class_name, raw_caption, background_prompt):
    caption = " ".join(raw_caption.split())
    caption = caption.strip("\"' ")

    if class_name.lower() not in caption.lower():
        caption = f"{class_name}, {caption}".strip(", ")

    if background_prompt.lower() not in caption.lower() and background_prompt != "plain background":
        caption = f"{caption}, {background_prompt}"

    return caption


def main():
    args = parse_args()
    os.makedirs(args.output_base_dir, exist_ok=True)

    try:
        class_map = load_dataset_classes(args)
    except Exception as exc:
        print(exc)
        return

    try:
        raw_labels = load_raw_labels(args.label_file)
    except FileNotFoundError:
        print(f"{args.label_file} not found")
        return

    image_files = load_image_files(args.image_dir)

    print(f"Loading Qwen3-VL from {args.model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    context_path = os.path.join(args.output_base_dir, args.context_filename)
    foreground_path = os.path.join(args.output_base_dir, args.foreground_filename)
    background_path = os.path.join(args.output_base_dir, args.background_filename)

    context_file = open(context_path, "w", encoding="utf-8")
    foreground_file = open(foreground_path, "w", encoding="utf-8")
    background_file = open(background_path, "w", encoding="utf-8")

    for idx, img_name in enumerate(tqdm(image_files, desc="Processing Qwen", unit="img")):
        img_path = os.path.join(args.image_dir, img_name)

        class_name = "object"
        if idx < len(raw_labels):
            label_id = raw_labels[idx] - 1
            class_name = class_map.get(label_id, "object")

        try:
            image = Image.open(img_path).convert("RGB")

            raw_context_prompt = run_qwen_caption(
                model,
                processor,
                image,
                build_context_instruction(class_name),
                max_new_tokens=15,
            )
            raw_background_prompt = run_qwen_caption(
                model,
                processor,
                image,
                build_background_instruction(class_name),
                max_new_tokens=8,
            )

            foreground_prompt = class_name
            background_prompt = sanitize_background_prompt(class_name, raw_background_prompt)
            context_prompt = sanitize_context_prompt(class_name, raw_context_prompt, background_prompt)

            context_file.write(f"{context_prompt}\n")
            foreground_file.write(f"{foreground_prompt}\n")
            background_file.write(f"{background_prompt}\n")

            context_file.flush()
            foreground_file.flush()
            background_file.flush()

        except Exception as exc:
            print(f"!!! Error processing {img_name}: {exc}")
            context_file.write(f"{class_name}\n")
            foreground_file.write(f"{class_name}\n")
            background_file.write("plain background\n")

            context_file.flush()
            foreground_file.flush()
            background_file.flush()

    context_file.close()
    foreground_file.close()
    background_file.close()

    print("Qwen 15-token prompt triplet generation finished.")


if __name__ == "__main__":
    main()
