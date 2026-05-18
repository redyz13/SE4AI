import argparse
import json
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.models import load_model


GENERATOR_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

TASK_TAXONOMY_PATH = PROJECT_ROOT / "outputs" / "taxonomy" / "final_task_taxonomy.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "generated" / "base_scenarios.jsonl"
SCENARIO_TEMPLATE_DIR = PROJECT_ROOT / "data" / "prompts" / "scenario_generation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Scenario generation template not found: {path}")

    return path.read_text(encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_template(template: str, values: dict) -> str:
    rendered = template

    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

    return rendered


def extract_json_array(text: str) -> list[dict]:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict) and "scenarios" in parsed:
            return parsed["scenarios"]

    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)

    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON array from model output:\n{text}")


def is_valid_scenario(scenario: dict) -> bool:
    if not isinstance(scenario, dict):
        return False

    scenario_text = scenario.get("scenario_text")
    task_instruction = scenario.get("task_instruction")

    if not isinstance(scenario_text, str) or not scenario_text.strip():
        return False

    if not isinstance(task_instruction, str) or not task_instruction.strip():
        return False

    return True


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_duplicate_scenario(
    scenario: dict,
    existing_scenarios: list[dict],
) -> bool:
    scenario_text = normalize_text(scenario.get("scenario_text", ""))
    task_instruction = normalize_text(scenario.get("task_instruction", ""))

    for existing in existing_scenarios:
        existing_text = normalize_text(existing.get("scenario_text", ""))
        existing_instruction = normalize_text(existing.get("task_instruction", ""))

        if scenario_text == existing_text and task_instruction == existing_instruction:
            return True

    return False


def get_scenario_template_path(task: str, subtask: str) -> Path:
    return SCENARIO_TEMPLATE_DIR / task / f"{subtask}.txt"


def build_generator_prompt(
    template: str,
    task: str,
    subtask: str,
    n: int,
    attempt: int,
) -> str:
    base_prompt = render_template(
        template,
        {
            "n": n,
            "task": task,
            "subtask": subtask,
        },
    )

    return (
        base_prompt
        + "\n\nAdditional generation instruction:\n"
        + f"- This is generation attempt {attempt}. Produce a scenario that is different from previous possible scenarios for the same subtask, while still following all constraints.\n"
    )


def iter_subtasks(task_taxonomy: dict):
    tasks = task_taxonomy["tasks"]

    for task_name, task_data in tasks.items():
        for subtask_name in task_data["subtasks"].keys():
            yield task_name, subtask_name


def get_model_device(model):
    return next(model.parameters()).device


def run_generation_prompt(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    device = get_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def generate_single_scenario(
    tokenizer,
    model,
    template: str,
    task: str,
    subtask: str,
    attempt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    prompt = build_generator_prompt(
        template=template,
        task=task,
        subtask=subtask,
        n=1,
        attempt=attempt,
    )

    raw_output = run_generation_prompt(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    scenarios = extract_json_array(raw_output)

    if len(scenarios) != 1:
        print(
            f"Warning: expected 1 scenario for {task}.{subtask}, "
            f"got {len(scenarios)} on attempt {attempt}."
        )

    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Generate neutral base scenarios for SE4AI prompt-pair generation."
    )

    parser.add_argument(
        "--scenarios-per-subtask",
        type=int,
        default=2,
        help=(
            "Number of neutral scenarios to generate for each subtask. "
            "Each scenario is generated with a separate model call."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where generated base scenarios are saved.",
    )

    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=5,
        help=(
            "Maximum number of generation attempts per subtask, expressed as a multiplier "
            "of --scenarios-per-subtask."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens generated for each scenario-generation call.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used only for scenario generation.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value used only for scenario generation.",
    )

    args = parser.parse_args()

    task_taxonomy = load_json(TASK_TAXONOMY_PATH)

    tokenizer, model, _ = load_model(GENERATOR_MODEL_ID)

    rows = []

    for task, subtask in iter_subtasks(task_taxonomy):
        print(f"\nGenerating scenarios for {task}.{subtask}")

        template_path = get_scenario_template_path(task, subtask)
        template = load_text(template_path)

        generated_count = 0
        attempt = 0
        max_attempts = args.scenarios_per_subtask * args.max_attempts_multiplier
        subtask_scenarios = []

        while generated_count < args.scenarios_per_subtask and attempt < max_attempts:
            attempt += 1

            try:
                scenarios = generate_single_scenario(
                    tokenizer=tokenizer,
                    model=model,
                    template=template,
                    task=task,
                    subtask=subtask,
                    attempt=attempt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as error:
                print(f"Skipping failed generation for {task}.{subtask} on attempt {attempt}: {error}")
                continue

            for scenario in scenarios:
                if generated_count >= args.scenarios_per_subtask:
                    break

                if not is_valid_scenario(scenario):
                    print(f"Skipping malformed scenario for {task}.{subtask}: {scenario}")
                    continue

                if is_duplicate_scenario(scenario, subtask_scenarios):
                    print(f"Skipping duplicate scenario for {task}.{subtask} on attempt {attempt}.")
                    continue

                generated_count += 1
                subtask_scenarios.append(scenario)

                rows.append(
                    {
                        "scenario_id": f"{task}_{subtask}_{generated_count:04d}",
                        "task": task,
                        "subtask": subtask,
                        "scenario_text": scenario["scenario_text"].strip(),
                        "task_instruction": scenario["task_instruction"].strip(),
                        "generator_model": GENERATOR_MODEL_ID,
                    }
                )

        if generated_count < args.scenarios_per_subtask:
            print(
                f"Warning: generated only {generated_count}/"
                f"{args.scenarios_per_subtask} scenarios for {task}.{subtask}."
            )

    output_path = Path(args.output)
    write_jsonl(output_path, rows)

    print(f"\nSaved {len(rows)} scenarios to {output_path}")


if __name__ == "__main__":
    main()