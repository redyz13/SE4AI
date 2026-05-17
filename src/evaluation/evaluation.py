import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch

from llm.inference import run_prompt
from llm.models import load_model


def load_prompt_pairs(input_path: Path) -> list[dict]:
    rows = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def extract_json(text: str) -> dict:
    cleaned = text.strip()

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if not isinstance(data, dict):
        return {
            "label": None,
            "confidence": None,
            "reason": "[INVALID_JSON_OR_THINKING_OUTPUT]",
        }

    reason = data.get("reason")

    if isinstance(reason, str):
        words = reason.split()
        if len(words) > 30:
            data["reason"] = " ".join(words[:30])

    return data


def compute_metrics(original_output: dict, counterfactual_output: dict) -> dict:
    original_label = original_output.get("label")
    counterfactual_label = counterfactual_output.get("label")

    original_confidence = original_output.get("confidence")
    counterfactual_confidence = counterfactual_output.get("confidence")

    label_flip = None
    if original_label is not None and counterfactual_label is not None:
        label_flip = original_label != counterfactual_label

    confidence_shift = None
    if isinstance(original_confidence, int) and isinstance(counterfactual_confidence, int):
        confidence_shift = abs(original_confidence - counterfactual_confidence)

    return {
        "label_flip": label_flip,
        "confidence_shift": confidence_shift,
    }


def run_evaluation(
    model_key: str,
    model_id: str,
    project_root: Path,
    input_path: Path,
    output_dir: Path,
    max_new_tokens: int = 180,
) -> Path:
    tokenizer, model, backend = load_model(model_id)
    prompt_pairs = load_prompt_pairs(input_path)

    results = []

    for index, pair in enumerate(prompt_pairs, start=1):
        print(f"\nRunning pair {index}/{len(prompt_pairs)}")
        print(f"{pair['original_term']} -> {pair['counterfactual_term']}")

        original_text = run_prompt(
            tokenizer=tokenizer,
            model=model,
            prompt=pair["original_prompt"],
            max_new_tokens=max_new_tokens,
        )

        counterfactual_text = run_prompt(
            tokenizer=tokenizer,
            model=model,
            prompt=pair["counterfactual_prompt"],
            max_new_tokens=max_new_tokens,
        )

        original_output = extract_json(original_text)
        counterfactual_output = extract_json(counterfactual_text)

        metrics = compute_metrics(original_output, counterfactual_output)

        results.append(
            {
                "model_key": model_key,
                "model_id": model_id,
                "backend": backend,
                "pair_id": pair.get("pair_id"),
                "scenario_id": pair.get("scenario_id"),
                "task": pair["task"],
                "subtask": pair["subtask"],
                "bias_axis": pair["bias_axis"],
                "replacement_group": pair["replacement_group"],
                "original_value": pair.get("original_value"),
                "counterfactual_value": pair.get("counterfactual_value"),
                "original_term": pair["original_term"],
                "counterfactual_term": pair["counterfactual_term"],
                "original_raw_output": original_text,
                "counterfactual_raw_output": counterfactual_text,
                "original_label": original_output.get("label"),
                "counterfactual_label": counterfactual_output.get("label"),
                "original_confidence": original_output.get("confidence"),
                "counterfactual_confidence": counterfactual_output.get("confidence"),
                "original_reason": original_output.get("reason"),
                "counterfactual_reason": counterfactual_output.get("reason"),
                "label_flip": metrics["label_flip"],
                "confidence_shift": metrics["confidence_shift"],
            }
        )

        print("Original:", original_output)
        print("Counterfactual:", counterfactual_output)
        print("Metrics:", metrics)

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_id.replace("/", "__")
    output_path = output_dir / f"results_{model_key}_{safe_model_name}.csv"

    pd.DataFrame(results).to_csv(output_path, index=False)

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nSaved results to: {output_path}")

    return output_path