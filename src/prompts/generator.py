import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_template(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8")


def render_template(template: str, values: dict) -> str:
    rendered = template

    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

    return rendered


def get_template_path(project_root: Path, task: str, subtask: str) -> Path:
    return project_root / "data" / "prompts" / task / f"{subtask}.txt"


def flatten_replacement_group(
    replacement_taxonomy: dict,
    axis: str,
    group: str,
) -> dict[str, list[str]]:
    group_data = replacement_taxonomy["replacement_taxonomy"][axis][group]
    return group_data["values"]


def build_term_pairs(
    values: dict[str, list[str]],
    max_pairs: int,
) -> list[tuple[str, str, str, str]]:
    value_keys = list(values.keys())
    pairs = []

    for i in range(len(value_keys)):
        for j in range(i + 1, len(value_keys)):
            left_key = value_keys[i]
            right_key = value_keys[j]

            if not values[left_key] or not values[right_key]:
                continue

            left_term = values[left_key][0]
            right_term = values[right_key][0]

            pairs.append((left_key, right_key, left_term, right_term))

            if len(pairs) >= max_pairs:
                return pairs

    return pairs


def render_sensitive_sentence(
    task: str,
    subtask: str,
    axis: str,
    group: str,
    term: str,
) -> str:
    if group in {"person_name", "proxy_name"}:
        if task == "decision_answering":
            if subtask == "candidate_comparison":
                return f"Candidate A name: {term}"

            if subtask == "scholarship_allocation":
                return f"Applicant A name: {term}"

            if subtask == "ticket_escalation_decision":
                return f"Requester for Ticket A name: {term}"

            if subtask == "public_service_priority_decision":
                return f"Requester for Request A name: {term}"

            return f"Subject A name: {term}"

        if subtask == "hiring_screening":
            return f"Candidate name: {term}"

        if subtask == "student_evaluation":
            return f"Student name: {term}"

        if subtask == "ticket_priority_classification":
            return f"Requester name: {term}"

        if subtask == "eligibility_classification":
            return f"Applicant name: {term}"

        return f"Name: {term}"

    if axis == "gender":
        return f"Gender-related descriptor: {term}"

    if axis == "race_ethnicity":
        return f"Race/ethnicity descriptor: {term}"

    if axis == "nationality":
        return f"Nationality or origin descriptor: {term}"

    if axis == "religion":
        return f"Religious identity descriptor: {term}"

    if axis == "age":
        return f"Age descriptor: {term}"

    if axis == "disability":
        return f"Disability-related descriptor: {term}"

    if axis == "socioeconomic_status":
        return f"Socioeconomic background descriptor: {term}"

    if axis == "sexual_orientation":
        return f"Sexual orientation descriptor: {term}"

    if axis == "physical_appearance":
        return f"Physical appearance descriptor: {term}"

    if axis == "language_background":
        return f"Language background descriptor: {term}"

    return f"{axis}: {term}"


def build_prompt_pairs_from_taxonomy(
    project_root: Path,
    scenarios_path: Path,
    task_taxonomy_path: Path,
    replacement_taxonomy_path: Path,
    output_path: Path,
    max_pairs_per_group: int = 2,
) -> Path:
    scenarios = load_jsonl(scenarios_path)
    task_taxonomy = load_json(task_taxonomy_path)
    replacement_taxonomy = load_json(replacement_taxonomy_path)

    output_rows = []

    for scenario in scenarios:
        task = scenario["task"]
        subtask = scenario["subtask"]

        template_path = get_template_path(project_root, task, subtask)

        if not template_path.exists():
            print(f"Skipping {task}.{subtask}: missing template {template_path}")
            continue

        template = load_template(template_path)

        allowed_axes = task_taxonomy["tasks"][task]["subtasks"][subtask]

        for axis, groups in allowed_axes.items():
            for group in groups:
                if axis not in replacement_taxonomy["replacement_taxonomy"]:
                    continue

                if group not in replacement_taxonomy["replacement_taxonomy"][axis]:
                    continue

                values = flatten_replacement_group(
                    replacement_taxonomy=replacement_taxonomy,
                    axis=axis,
                    group=group,
                )

                term_pairs = build_term_pairs(
                    values=values,
                    max_pairs=max_pairs_per_group,
                )

                for pair_index, (
                    original_value,
                    counterfactual_value,
                    original_term,
                    counterfactual_term,
                ) in enumerate(term_pairs, start=1):
                    original_sensitive_sentence = render_sensitive_sentence(
                        task=task,
                        subtask=subtask,
                        axis=axis,
                        group=group,
                        term=original_term,
                    )

                    counterfactual_sensitive_sentence = render_sensitive_sentence(
                        task=task,
                        subtask=subtask,
                        axis=axis,
                        group=group,
                        term=counterfactual_term,
                    )

                    base_values = {
                        "scenario_text": scenario["scenario_text"],
                        "task_instruction": scenario["task_instruction"],
                    }

                    original_prompt = render_template(
                        template,
                        {
                            **base_values,
                            "sensitive_sentence": original_sensitive_sentence,
                        },
                    )

                    counterfactual_prompt = render_template(
                        template,
                        {
                            **base_values,
                            "sensitive_sentence": counterfactual_sensitive_sentence,
                        },
                    )

                    pair_id = (
                        f"{scenario['scenario_id']}_"
                        f"{axis}_{group}_{pair_index:04d}"
                    )

                    output_rows.append(
                        {
                            "pair_id": pair_id,
                            "scenario_id": scenario["scenario_id"],
                            "task": task,
                            "subtask": subtask,
                            "bias_axis": axis,
                            "replacement_group": group,
                            "original_value": original_value,
                            "counterfactual_value": counterfactual_value,
                            "original_term": original_term,
                            "counterfactual_term": counterfactual_term,
                            "original_prompt": original_prompt,
                            "counterfactual_prompt": counterfactual_prompt,
                        }
                    )

    write_jsonl(output_path, output_rows)

    print(f"\nSaved {len(output_rows)} prompt pairs to {output_path}")

    return output_path