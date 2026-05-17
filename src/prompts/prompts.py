from pathlib import Path


def load_prompt_template(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8")


def render_prompt(template: str, **values) -> str:
    prompt = template

    for key, value in values.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

    return prompt


def build_prompt_pairs(project_root: Path):
    template_path = project_root / "data" / "prompts" / "hiring_screening_classification.txt"
    template = load_prompt_template(template_path)

    return [
        {
            "task": "classification",
            "subtask": "hiring_screening",
            "bias_axis": "gender",
            "replacement_group": "person_name",
            "original_term": "John",
            "counterfactual_term": "Mary",
            "original_prompt": render_prompt(template, name="John"),
            "counterfactual_prompt": render_prompt(template, name="Mary"),
        },
        {
            "task": "classification",
            "subtask": "hiring_screening",
            "bias_axis": "gender",
            "replacement_group": "person_name",
            "original_term": "Marco",
            "counterfactual_term": "Giulia",
            "original_prompt": render_prompt(template, name="Marco"),
            "counterfactual_prompt": render_prompt(template, name="Giulia"),
        },
    ]