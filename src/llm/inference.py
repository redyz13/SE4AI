import re
import torch


def is_qwen3_model(model) -> bool:
    config = getattr(model, "config", None)

    model_type = str(getattr(config, "model_type", "")).lower()
    model_name = str(getattr(config, "_name_or_path", "")).lower()
    class_name = model.__class__.__name__.lower()

    return "qwen3" in model_type or "qwen3" in model_name or "qwen3" in class_name


def get_input_device(model):
    if hasattr(model, "hf_device_map"):
        for _, device in model.hf_device_map.items():
            if device not in ["cpu", "disk"]:
                return torch.device(device)
        return torch.device("cpu")

    return next(model.parameters()).device


def remove_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Caso in cui Qwen apre <think> ma non arriva mai a chiuderlo
    if text.startswith("<think>"):
        return ""

    return text


def run_prompt(tokenizer, model, prompt: str, max_new_tokens: int = 180) -> str:
    qwen3 = is_qwen3_model(model)

    if qwen3:
        prompt = prompt + "\n\n/no_think"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if qwen3:
        chat_template_kwargs["enable_thinking"] = False

    try:
        text = tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs,
        )
    except TypeError:
        # Fallback per tokenizer che non supportano enable_thinking
        chat_template_kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs,
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    device = get_input_device(model)

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return remove_thinking_blocks(generated_text)