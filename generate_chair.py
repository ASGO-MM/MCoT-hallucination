from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_messages(image_path: str, question: str) -> list[dict]:

    prompt_suffix = (
        "You FIRST think about the reasoning process as an internal monologue and "
        "then provide the final answer. The reasoning process MUST BE enclosed within "
        "<think> </think> tags. The final answer MUST BE in <answer> </answer> tags."
    )


    text = f"Question: {question}{prompt_suffix}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    ]


def resolve_image_path(image_root: str, image_id: str) -> str:
    if os.path.isabs(image_id):
        return image_id
    return os.path.join(image_root, image_id)


def load_records(input_path: str) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{input_path} must contain a JSON list")
    return records


def randomize_records(records: list[dict]) -> list[dict]:
    shuffled = list(records)
    random.SystemRandom().shuffle(shuffled)
    return shuffled


def select_chunk(records: list[dict], num_chunks: int, chunk_index: int, num_samples: int | None) -> list[dict]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError("chunk_index must satisfy 0 <= chunk_index < num_chunks")

    if num_chunks > 1:
        chunk_size = (len(records) + num_chunks - 1) // num_chunks
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(records))
        records = records[start:end]

    if num_samples is not None:
        records = records[:num_samples]

    return records


def load_model_and_processor(model_id: str, attn_implementation: str, device: str):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    ).to(device).eval()
    return model, processor


def prepare_inputs(processor, image_path: str, question: str, device: str) -> dict[str, torch.Tensor]:
    messages = build_messages(image_path, question)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items()}


def model_inputs_for_step(base_inputs: dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    step_inputs = dict(base_inputs)
    step_inputs["input_ids"] = input_ids
    step_inputs["attention_mask"] = attention_mask
    return step_inputs


def get_eos_token_ids(tokenizer) -> set[int]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


class ActivationContext:
    def __init__(self, entropy_vec: torch.Tensor, baseline_logits: torch.Tensor):
        self.entropy_vec = entropy_vec
        self.baseline_logits = baseline_logits


def compute_activation_context(
    model,
    base_inputs: dict[str, torch.Tensor],
    activation_info_layer: int,
) -> ActivationContext:
    with torch.no_grad():
        outputs = model(
            **base_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model forward did not return hidden states")

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("model.config.image_token_id is required for visual-only entropy")

    input_ids = base_inputs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("input_ids missing from processor output")

    visual_mask = input_ids[0] == int(image_token_id)
    if not torch.any(visual_mask):
        raise RuntimeError("no visual placeholder tokens were found in the prompt")

    layer_hidden = hidden_states[activation_info_layer][0]
    visual_hidden = layer_hidden[visual_mask]
    visual_count = int(visual_hidden.shape[0])
    if visual_count <= 1:
        raise RuntimeError("at least two visual positions are required to compute normalized entropy")

    lm_head = model.get_output_embeddings()
    lm_weight = lm_head.weight
    lm_bias = getattr(lm_head, "bias", None)

    token_position_logits = visual_hidden @ lm_weight.T
    if lm_bias is not None:
        token_position_logits = token_position_logits + lm_bias

    token_position_logits = token_position_logits.transpose(0, 1).float()
    position_probs = torch.softmax(token_position_logits, dim=-1)
    entropy_vec = -(position_probs * torch.log(position_probs.clamp_min(1e-20))).sum(dim=-1)
    entropy_vec = entropy_vec / math.log(visual_count)

    return ActivationContext(
        entropy_vec=entropy_vec,
        baseline_logits=outputs.logits[:, -1, :],
    )


def maybe_apply_activation(
    step_logits: torch.Tensor,
    entropy_vec: torch.Tensor,
    activation_alpha: float | None,
    activation_threshold: float | None,
) -> tuple[torch.Tensor, dict]:
    if activation_alpha is None:
        current_token_id = int(step_logits.argmax(dim=-1).item())
        current_entropy = float(entropy_vec[current_token_id].item())
        return step_logits, {
            "current_token_id": current_token_id,
            "current_token_entropy": current_entropy,
            "gate_applied": False,
        }

    current_token_id = int(step_logits.argmax(dim=-1).item())
    current_entropy = float(entropy_vec[current_token_id].item())
    should_intervene = activation_threshold is None or current_entropy > activation_threshold

    if should_intervene:
        adjusted_logits = step_logits - activation_alpha * entropy_vec.unsqueeze(0).to(step_logits.dtype)
    else:
        adjusted_logits = step_logits

    return adjusted_logits, {
        "current_token_id": current_token_id,
        "current_token_entropy": current_entropy,
        "gate_applied": bool(should_intervene),
    }


def greedy_generate(
    model,
    processor,
    base_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    activation_alpha: float | None,
    activation_info_layer: int,
    activation_threshold: float | None,
) -> tuple[str, list[dict]]:
    current_input_ids = base_inputs["input_ids"].clone()
    current_attention_mask = base_inputs["attention_mask"].clone()
    eos_token_ids = get_eos_token_ids(processor.tokenizer)

    generated_token_ids: list[int] = []
    gate_trace: list[dict] = []
    if activation_alpha is not None:
        activation_context = compute_activation_context(
            model=model,
            base_inputs=base_inputs,
            activation_info_layer=activation_info_layer,
        )
        step_logits = activation_context.baseline_logits
    else:
        activation_context = None
        with torch.no_grad():
            outputs = model(
                **base_inputs,
                use_cache=False,
                return_dict=True,
            )
        step_logits = outputs.logits[:, -1, :]

    for step_idx in range(max_new_tokens):
        if activation_context is not None:
            adjusted_logits, gate_meta = maybe_apply_activation(
                step_logits=step_logits,
                entropy_vec=activation_context.entropy_vec,
                activation_alpha=activation_alpha,
                activation_threshold=activation_threshold,
            )
        else:
            adjusted_logits = step_logits
            current_token_id = int(step_logits.argmax(dim=-1).item())
            gate_meta = {
                "current_token_id": current_token_id,
                "current_token_entropy": None,
                "gate_applied": False,
            }
        next_token_id = int(adjusted_logits.argmax(dim=-1).item())
        generated_token_ids.append(next_token_id)

        gate_trace.append(
            {
                "step": step_idx,
                "current_token_id": gate_meta["current_token_id"],
                "current_token": decode_token(processor.tokenizer, gate_meta["current_token_id"]),
                "current_token_entropy": gate_meta["current_token_entropy"],
                "gate_applied": gate_meta["gate_applied"],
                "selected_token_id": next_token_id,
                "selected_token": decode_token(processor.tokenizer, next_token_id),
            }
        )

        if next_token_id in eos_token_ids:
            break

        next_token = torch.tensor([[next_token_id]], device=current_input_ids.device, dtype=current_input_ids.dtype)
        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
        next_mask = torch.ones((1, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype)
        current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=-1)

        with torch.no_grad():
            outputs = model(
                **model_inputs_for_step(base_inputs, current_input_ids, current_attention_mask),
                use_cache=False,
                return_dict=True,
            )
        step_logits = outputs.logits[:, -1, :]

    generated_tensor = torch.tensor([generated_token_ids], device=current_input_ids.device)
    text = processor.batch_decode(
        generated_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text, gate_trace


def process_records(
    model,
    processor,
    records: list[dict],
    image_root: str,
    output_path: str,
    max_new_tokens: int,
    activation_alpha: float | None,
    activation_info_layer: int,
    activation_threshold: float | None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    outputs: list[dict] = []

    progress = tqdm(records, total=len(records), desc="Generating CHAIR captions")
    for idx, record in enumerate(progress, start=1):
        image_id = record.get("image_id")
        if not image_id:
            raise KeyError("each record must contain image_id")

        question = record.get("instruction", "")
        image_path = resolve_image_path(image_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image not found: {image_path}")

        base_inputs = prepare_inputs(
            processor=processor,
            image_path=image_path,
            question=question,
            device=str(model.device),
        )
        generated_text, gate_trace = greedy_generate(
            model=model,
            processor=processor,
            base_inputs=base_inputs,
            max_new_tokens=max_new_tokens,
            activation_alpha=activation_alpha,
            activation_info_layer=activation_info_layer,
            activation_threshold=activation_threshold,
        )

        output_record = dict(record)
        output_record["image_src"] = image_path
        output_record["question"] = question
        output_record["model_answer"] = generated_text
        output_record["activation_gate_trace"] = gate_trace
        outputs.append(output_record)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2, ensure_ascii=False)

        progress.set_postfix_str(image_id)
        print(f"[{idx}/{len(records)}] {image_id}")
        print(generated_text, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CHAIR input JSON with image_id and instruction fields")
    parser.add_argument("--output", type=str, required=True, help="Output JSON with model responses")
    parser.add_argument("--model_id", type=str, required=True, help="Local path or HF id for GRIT-3B")
    parser.add_argument("--image_root", type=str, required=True, help="Directory containing COCO images")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=0, help="Use <= 0 to process all records")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--activation_alpha", type=float, default=None, help="Set to enable activation decoding")
    parser.add_argument("--activation_info_layer", type=int, default=-1)
    parser.add_argument("--activation_threshold", type=float, default=0.5, help="Gamma in the paper")
    parser.add_argument("--disable_activation_gate", type=str2bool, default=False, help="If true, always apply Eq. 3 when activation is enabled")
    args = parser.parse_args()

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        device=args.device,
    )

    num_samples = args.num_samples if args.num_samples > 0 else None
    records = randomize_records(load_records(args.input))
    records = select_chunk(
        records=records,
        num_chunks=args.num_chunks,
        chunk_index=args.chunk_index,
        num_samples=num_samples,
    )
    if not records:
        raise ValueError("no records selected after chunking / sampling")

    activation_threshold = None if args.disable_activation_gate else args.activation_threshold

    process_records(
        model=model,
        processor=processor,
        records=records,
        image_root=args.image_root,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        activation_alpha=args.activation_alpha,
        activation_info_layer=args.activation_info_layer,
        activation_threshold=activation_threshold,
    )


if __name__ == "__main__":
    main()
