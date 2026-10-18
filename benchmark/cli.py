from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .evaluation import evaluate_scores, load_standardized_outputs
from .pipeline import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="RegArch simulation benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--edge-universe", type=Path, required=True)
    evaluate_parser.add_argument("--scores", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "generate":
        config = load_config(args.config)
        for replicate in range(1, config.replicates + 1):
            generated = generate_dataset(config, args.output, replicate)
            print(generated)
    elif args.command == "evaluate":
        import pandas as pd

        edge_universe = pd.read_csv(args.edge_universe, sep="\t")
        scores = load_standardized_outputs(args.scores)
        metrics = evaluate_scores(edge_universe, scores)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
