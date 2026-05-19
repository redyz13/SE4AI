"""
Run the full SE4AI counterfactual fairness pipeline.

This script can:
1. generate neutral base scenarios;
2. generate original/counterfactual prompt pairs from the taxonomy;
3. evaluate one or more target LLMs on the generated prompt pairs.

Example runs:

Run the full pipeline on Qwen with a small test configuration:
    python scripts/run_pipeline.py --models qwen --scenarios-per-subtask 1 --max-pairs-per-group 1 --max-new-tokens 120

Run the full pipeline on all target models:
    python scripts/run_pipeline.py --models qwen mistral llama --scenarios-per-subtask 1 --max-pairs-per-group 1 --max-new-tokens 120

Reuse existing base scenarios and regenerate only prompt pairs before evaluation:
    python scripts/run_pipeline.py --models qwen --skip-scenario-generation --max-pairs-per-group 1

Reuse existing prompt pairs and only run model evaluation:
    python scripts/run_pipeline.py --models qwen mistral llama --skip-scenario-generation --skip-prompt-generation

Use custom output paths:
    python scripts/run_pipeline.py --models qwen --scenarios-output data/generated/base_scenarios_test.jsonl --prompt-pairs-output data/generated/prompt_pairs_test.jsonl --output-dir outputs/llm_runs

Run evaluation with 4-bit quantization:
    python scripts/run_pipeline.py --models qwen mistral llama --skip-scenario-generation --skip-prompt-generation --quantization 4bit

Run evaluation with 8-bit quantization:
    python scripts/run_pipeline.py --models qwen mistral llama --skip-scenario-generation --skip-prompt-generation --quantization 8bit
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "data" / "generated" / "base_scenarios.jsonl"
DEFAULT_PROMPT_PAIRS_PATH = PROJECT_ROOT / "data" / "generated" / "prompt_pairs.jsonl"


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(command))
    print("=" * 80 + "\n")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{' '.join(command)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full SE4AI counterfactual fairness pipeline."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen"],
        choices=["qwen", "mistral", "llama"],
        help="Target models to evaluate.",
    )

    parser.add_argument(
        "--scenarios-per-subtask",
        type=int,
        default=1,
        help="Number of neutral base scenarios generated for each subtask.",
    )

    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=1,
        help="Maximum number of counterfactual pairs generated for each replacement group.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Maximum number of generated tokens for each evaluated prompt.",
    )

    parser.add_argument(
        "--scenarios-output",
        default=str(DEFAULT_SCENARIOS_PATH),
        help="Path where generated base scenarios are saved.",
    )

    parser.add_argument(
        "--prompt-pairs-output",
        default=str(DEFAULT_PROMPT_PAIRS_PATH),
        help="Path where generated prompt pairs are saved.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/llm_runs",
        help="Directory where model evaluation CSV files are saved.",
    )

    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode used for target model evaluation.",
    )

    parser.add_argument(
        "--skip-scenario-generation",
        action="store_true",
        help="Skip base scenario generation and reuse the existing scenarios file.",
    )

    parser.add_argument(
        "--skip-prompt-generation",
        action="store_true",
        help="Skip prompt pair generation and reuse the existing prompt pairs file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    python_executable = sys.executable

    scenarios_path = Path(args.scenarios_output)
    prompt_pairs_path = Path(args.prompt_pairs_output)

    if not args.skip_scenario_generation:
        run_command(
            [
                python_executable,
                "scripts/generate_base_scenarios.py",
                "--scenarios-per-subtask",
                str(args.scenarios_per_subtask),
                "--output",
                str(scenarios_path),
            ]
        )
    else:
        print(f"Skipping scenario generation. Reusing: {scenarios_path}")

    if not args.skip_prompt_generation:
        run_command(
            [
                python_executable,
                "scripts/generate_prompt_pairs.py",
                "--scenarios",
                str(scenarios_path),
                "--output",
                str(prompt_pairs_path),
                "--max-pairs-per-group",
                str(args.max_pairs_per_group),
            ]
        )
    else:
        print(f"Skipping prompt generation. Reusing: {prompt_pairs_path}")

    if not prompt_pairs_path.exists():
        raise FileNotFoundError(
            f"Prompt pairs file not found: {prompt_pairs_path}"
        )

    for model in args.models:
        run_command(
            [
                python_executable,
                "scripts/run_llm_eval.py",
                "--model",
                model,
                "--input",
                str(prompt_pairs_path),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--output-dir",
                args.output_dir,
                "--quantization",
                args.quantization,
            ]
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()