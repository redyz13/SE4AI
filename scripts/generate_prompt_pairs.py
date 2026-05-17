import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prompts.generator import build_prompt_pairs_from_taxonomy


DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "data" / "generated" / "base_scenarios.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "generated" / "prompt_pairs.jsonl"
TASK_TAXONOMY_PATH = PROJECT_ROOT / "outputs" / "taxonomy" / "final_task_taxonomy.json"
REPLACEMENT_TAXONOMY_PATH = PROJECT_ROOT / "outputs" / "taxonomy" / "final_replacement_taxonomy.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-pairs-per-group", type=int, default=2)

    args = parser.parse_args()

    build_prompt_pairs_from_taxonomy(
        project_root=PROJECT_ROOT,
        scenarios_path=Path(args.scenarios),
        task_taxonomy_path=TASK_TAXONOMY_PATH,
        replacement_taxonomy_path=REPLACEMENT_TAXONOMY_PATH,
        output_path=Path(args.output),
        max_pairs_per_group=args.max_pairs_per_group,
    )


if __name__ == "__main__":
    main()