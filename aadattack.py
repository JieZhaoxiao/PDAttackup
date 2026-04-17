import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from tqdm import tqdm

import attack_base
import other_attacks
from attention import AttentionStore
from utils import aggregate_attention, view_images

preprocess = attack_base.preprocess
ddim_reverse_sample = attack_base.ddim_reverse_sample
register_attention_control = attack_base.register_attention_control
reset_attention_control = attack_base.reset_attention_control
init_latent = attack_base.init_latent
diffusion_step = attack_base.diffusion_step
latent2image = attack_base.latent2image


def load_dataset_labels(args):
    if args.dataset_name == "imagenet_compatible":
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError
    return imagenet_label


def load_prompt_lines(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_lines = file.readlines()

    if len(prompt_lines) == 0:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    return prompt_lines


def sanitize_background_prompt(background_prompt, label_name):
    background_prompt = " ".join(background_prompt.split())
    background_prompt = re.compile(re.escape(label_name), re.IGNORECASE).sub("", background_prompt)
    background_prompt = re.sub(r"[^a-zA-Z0-9, ]+", " ", background_prompt)
    background_prompt = re.sub(r"\s*,\s*", ", ", background_prompt)
    background_prompt = " ".join(background_prompt.split()).strip(", ")

    if not background_prompt:
        background_prompt = "plain background"

    banned_terms = {
        "person", "people", "man", "woman", "boy", "girl", "child", "children",
        "hand", "hands", "leg", "legs", "arm", "arms", "face", "head",
        "dog", "cat", "horse", "bird", "player", "crowd", "husky",
        "shirt", "jacket", "coat", "pants", "shoe", "shoes", "glove", "gloves", "helmet", "hat", "cap",
    }

    segments = []
    for segment in background_prompt.split(","):
        segment = " ".join(segment.split()).strip()
        if not segment:
            continue

        words = segment.split()[:3]
        if any(word.lower() in banned_terms for word in words):
            continue

        segment = " ".join(words).strip(", ")
        if segment and segment not in segments:
            segments.append(segment)

    if not segments:
        segments = ["plain background"]

    return ", ".join(segments[:1])


def build_context_schedule(model, all_uncond_emb, prompt_text, batch_size):
    prompt_batch = [prompt_text] * batch_size
    text_input = model.tokenizer(
        prompt_batch,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [
        [torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings]
        for i in range(len(all_uncond_emb))
    ]
    context = [torch.cat(item) for item in context]
    return prompt_batch, context


def run_context_branch(model, controller, prompt_batch, context_schedule, original_latent, latent, start_step, guidance_scale):
    register_attention_control(model, controller)
    controller.loss = 0
    controller.reset()

    latents = torch.cat([original_latent, latent])
    for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
        latents = diffusion_step(model, latents, context_schedule[ind], t, guidance_scale)

    return latents, controller.loss


@torch.no_grad()
def compute_clean_attention_map(model, prompt_text, context_schedule, latent, start_step, guidance_scale, res):
    controller = AttentionStore(res)
    register_attention_control(model, controller)
    controller.reset()

    latents = latent.detach().clone()
    for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
        latents = diffusion_step(model, latents, context_schedule[ind], t, guidance_scale)

    attention_map = aggregate_attention(
        [prompt_text],
        controller,
        res // 32,
        ("up", "down"),
        True,
        0,
        is_cpu=False,
    )
    return attention_map.detach()


def compute_live_attention_map(model, prompt_text, context_schedule, latent, start_step, guidance_scale, res):
    controller = AttentionStore(res)
    register_attention_control(model, controller)
    controller.reset()

    latents = latent
    for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
        latents = diffusion_step(model, latents, context_schedule[ind], t, guidance_scale)

    attention_map = aggregate_attention(
        [prompt_text],
        controller,
        res // 32,
        ("up", "down"),
        True,
        0,
        is_cpu=False,
    )
    return attention_map


def extract_prompt_attention_map(attention_map, prompt_text, tokenizer):
    prompt_tokens = tokenizer.encode(prompt_text)
    valid_token_count = len(prompt_tokens) - 2
    if valid_token_count <= 0:
        raise ValueError(f"Prompt has no valid tokens: {prompt_text}")

    prompt_attention_map = attention_map[:, :, 1:1 + valid_token_count].mean(dim=2, keepdim=True)
    return normalize_attention_map(prompt_attention_map), prompt_tokens


def normalize_attention_map(attention_map):
    min_value = attention_map.amin(dim=(0, 1), keepdim=True)
    max_value = attention_map.amax(dim=(0, 1), keepdim=True)
    return (attention_map - min_value) / (max_value - min_value + 1e-8)


def build_complementary_attention(foreground_map, background_map):
    return normalize_attention_map(foreground_map * (1.0 - background_map))


def cosine_injection_loss(complementary_map, background_map):
    complementary_vector = complementary_map.reshape(1, -1)
    background_vector = background_map.reshape(1, -1)
    return 1.0 - F.cosine_similarity(complementary_vector, background_vector).mean()


def log_prompt_summary(img_index, context_prompt, foreground_prompt, background_prompt):
    print(f"\n[Prompt] Image {img_index} - Context: \"{context_prompt}\"")
    print(f"[Prompt] Image {img_index} - Foreground: \"{foreground_prompt}\"")
    print(f"[Prompt] Image {img_index} - Background: \"{background_prompt}\"")
    print("[Prompt] Complementary foreground map: fg * (1 - bg)\n")


@torch.enable_grad()
def diffattack(
    model,
    label,
    controller,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image=None,
    model_name="inception",
    save_path=r"C:\Users\PC\Desktop\output",
    res=224,
    start_step=15,
    iterations=30,
    verbose=True,
    topN=1,
    img_index=0,
    args=None,
):
    del topN

    imagenet_label = load_dataset_labels(args)
    context_prompt_lines = load_prompt_lines(args.context_prompt_path)
    foreground_prompt_lines = load_prompt_lines(args.foreground_prompt_path)
    background_prompt_lines = load_prompt_lines(args.background_prompt_path)

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)
    _ = classifier(test_image.cuda())

    label_name = imagenet_label.refined_Label[label.item()]
    context_prompt = context_prompt_lines[img_index].strip() or label_name
    foreground_prompt = foreground_prompt_lines[img_index].strip() or label_name
    background_prompt = sanitize_background_prompt(background_prompt_lines[img_index].strip(), label_name)

    log_prompt_summary(img_index, context_prompt, foreground_prompt, background_prompt)

    latent, inversion_latents = ddim_reverse_sample(
        image,
        [context_prompt],
        model,
        num_inference_steps,
        0,
        res=height,
    )
    inversion_latents = inversion_latents[::-1]
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context_text_input = model.tokenizer(
        [context_prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    context_text_embeddings = model.text_encoder(context_text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, 1)

    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=2e-1)
    loss_func = torch.nn.MSELoss()

    inversion_context = torch.cat([uncond_embeddings, context_text_embeddings])

    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, inversion_context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
            loss.backward()
            optimizer.step()
            inversion_context = torch.cat([uncond_embeddings, context_text_embeddings])

        with torch.no_grad():
            latents = diffusion_step(model, latents, inversion_context, t, guidance_scale).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    uncond_embeddings.requires_grad_(False)

    context_prompt_batch, context_schedule = build_context_schedule(model, all_uncond_emb, context_prompt, batch_size=2)
    _, foreground_context_single = build_context_schedule(model, all_uncond_emb, foreground_prompt, batch_size=1)
    _, background_context_single = build_context_schedule(model, all_uncond_emb, background_prompt, batch_size=1)

    original_latent = latent.clone().detach()
    original_latent_single = original_latent.clone()
    latent.requires_grad_(True)

    clean_foreground_attention = compute_clean_attention_map(
        model,
        foreground_prompt,
        foreground_context_single,
        original_latent_single,
        start_step,
        guidance_scale,
        args.res,
    )
    clean_background_attention = compute_clean_attention_map(
        model,
        background_prompt,
        background_context_single,
        original_latent_single,
        start_step,
        guidance_scale,
        args.res,
    )

    clean_foreground_map, foreground_tokens = extract_prompt_attention_map(
        clean_foreground_attention,
        foreground_prompt,
        model.tokenizer,
    )
    clean_background_map, background_tokens = extract_prompt_attention_map(
        clean_background_attention,
        background_prompt,
        model.tokenizer,
    )
    clean_complementary_map = build_complementary_attention(clean_foreground_map, clean_background_map)

    optimizer = optim.AdamW([latent], lr=args.latent_lr)
    cross_entro = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(iterations), desc="Iterations")
    for iteration_idx, _ in enumerate(pbar):
        context_latents, self_attn_loss = run_context_branch(
            model,
            controller,
            context_prompt_batch,
            context_schedule,
            original_latent,
            latent,
            start_step,
            guidance_scale,
        )

        live_foreground_attention = compute_live_attention_map(
            model,
            foreground_prompt,
            foreground_context_single,
            latent,
            start_step,
            guidance_scale,
            args.res,
        )
        live_background_attention = compute_live_attention_map(
            model,
            background_prompt,
            background_context_single,
            latent,
            start_step,
            guidance_scale,
            args.res,
        )

        live_foreground_map, _ = extract_prompt_attention_map(
            live_foreground_attention,
            foreground_prompt,
            model.tokenizer,
        )
        live_background_map, _ = extract_prompt_attention_map(
            live_background_attention,
            background_prompt,
            model.tokenizer,
        )
        live_complementary_map = build_complementary_attention(live_foreground_map, live_background_map)

        if iteration_idx == 0:
            decoded_foreground = model.tokenizer.decode(foreground_tokens[1:-1])
            decoded_background = model.tokenizer.decode(background_tokens[1:-1])
            print(
                "[Attention Summary] "
                f"Foreground tokens: `{decoded_foreground}` | "
                f"Background tokens: `{decoded_background}`"
            )

        init_out_image = model.vae.decode(1 / 0.18215 * context_latents)["sample"][1:]
        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        if args.dataset_name != "imagenet_compatible":
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)

        attack_loss = -cross_entro(pred, label) * args.attack_loss_weight
        transfer_loss = cosine_injection_loss(live_complementary_map, live_background_map) * args.transfer_loss_weight
        cross_attn_loss = F.mse_loss(live_complementary_map, clean_complementary_map.detach()) * args.cross_attn_loss_weight
        self_attn_loss = self_attn_loss * args.self_attn_loss_weight
        loss = attack_loss + transfer_loss + cross_attn_loss + self_attn_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix(
                attack=float(attack_loss.detach().item()),
                trans=float(transfer_loss.detach().item()),
                cross=float(cross_attn_loss.detach().item()),
                self_attn=float(self_attn_loss.detach().item()),
            )

    with torch.no_grad():
        register_attention_control(model, controller)
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context_schedule[ind], t, guidance_scale)

    image = latent2image(model.vae, latents.detach())
    perturbed = image[1:].astype(np.float32) / 255.0
    image = (perturbed * 255).astype(np.uint8)
    view_images(image, show=False, save_path=save_path + "_adv_image.png")
    reset_attention_control(model)

    return image[0], 0, 0
