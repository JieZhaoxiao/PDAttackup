import os
import torch
import json
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

LOCAL_MODEL_PATH = "Qwen3-VL-8B-Instruct"
image_dir = "imagenet_compatible/images"
label_file = "imagenet_compatible/labels.txt"
imagenet_json_path = "imagenet_compatible/imagenet_class_index.json"
output_base_dir = "Text/qwen3_vl"

os.makedirs(output_base_dir, exist_ok=True)

def load_imagenet_classes():
    if not os.path.exists(imagenet_json_path):
        raise FileNotFoundError(f"{imagenet_json_path}")
    with open(imagenet_json_path, "r", encoding='utf-8') as f:
        class_idx = json.load(f)
    return {int(k): v[1].replace('_', ' ') for k, v in class_idx.items()}

try:
    imagenet_map = load_imagenet_classes()
except Exception as e:
    print(e)
    exit()

try:
    with open(label_file, 'r', encoding='utf-8') as f:
        raw_labels = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
except FileNotFoundError:
    print(f"{label_file} not found")
    exit()

print(f"Loading Qwen3-VL from {LOCAL_MODEL_PATH}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

try:
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
except ValueError:
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])

buckets_stages = [
    {
        "name": "15_tokens",
        "instruction": "Write a caption for this image. The main subject MUST be '{class_name}'. If it is in a scene, mention one background object. If isolated, mention one prominent physical part. Keep the length under 15 tokens.",
        "max_new": 15
    },
    {
        "name": "45_tokens",
        "instruction": "Write a caption for this image. The main subject MUST be '{class_name}'. If it is in a scene, mention one background object. If isolated, mention one prominent physical part. Keep the length under 45 tokens.",
        "max_new": 45
    },
    {
        "name": "75_tokens",
        "instruction": "Write a caption for this image. The main subject MUST be '{class_name}'. If it is in a scene, mention one background object. If isolated, mention one prominent physical part. Keep the length under 75 tokens.",
        "max_new": 75
    }
]

file_handles = {}
for stage in buckets_stages:
    f_path = os.path.join(output_base_dir, f"{stage['name']}.txt")
    file_handles[stage['name']] = open(f_path, 'w', encoding='utf-8')

for idx, img_name in enumerate(tqdm(image_files, desc="Processing Qwen", unit="img")):
    img_path = os.path.join(image_dir, img_name)

    class_name = "object"
    if idx < len(raw_labels):
        label_id = raw_labels[idx] - 1
        class_name = imagenet_map.get(label_id, "object")

    try:
        image = Image.open(img_path).convert('RGB')

        for stage in buckets_stages:
            user_content = stage["instruction"].format(class_name=class_name)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_content},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=stage["max_new"],
                do_sample=False,
                repetition_penalty=1.1
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            clean_desc = " ".join(output_text.split())
            
            if class_name.lower() not in clean_desc.lower():
                clean_desc = f"{class_name}: {clean_desc}"

            file_handles[stage['name']].write(f"{clean_desc}\n")
            file_handles[stage['name']].flush()

    except Exception as e:
        print(f"!!! Error processing {img_name}: {e}")
        for fh in file_handles.values():
            fh.write(f"ERROR\n")
            fh.flush()

for fh in file_handles.values():
    fh.close()

print("Qwen Processing Done.")
