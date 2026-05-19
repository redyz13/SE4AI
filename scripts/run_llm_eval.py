import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from evaluation.evaluation import run_evaluation
from llm.models import MODEL_REGISTRY, get_model_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SE4AI counterfactual fairness evaluation on generated prompt pairs."
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Model to run.",
    )

    parser.add_argument(
        "--input",
        default="data/generated/prompt_pairs.jsonl",
        help="Path to the JSONL file containing original and counterfactual prompt pairs.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
        help="Maximum number of generated tokens for each prompt.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/llm_runs",
        help="Directory where CSV results are saved.",
    )

    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode used for model loading.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_id = get_model_id(args.model)

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prompt pairs file not found: {input_path}\n"
            "Generate it first with:\n"
            "python scripts/generate_prompt_pairs.py"
        )

    output_path = run_evaluation(
        model_key=args.model,
        model_id=model_id,
        project_root=PROJECT_ROOT,
        input_path=input_path,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
    )

    print(f"Done. Results saved at: {output_path}")


if __name__ == "__main__":
    main()