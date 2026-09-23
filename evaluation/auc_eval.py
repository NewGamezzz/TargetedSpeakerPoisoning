"""AUC between the retain and forget similarity distributions.

Takes the two CSVs written by ``ssim_eval.py`` and reports how separable the
retain-set and forget-set similarity scores are.

The retain set is the positive class, so the AUC is the probability that a
randomly drawn retain utterance scores higher than a randomly drawn forget one:

  0.5  the two distributions overlap completely -- no suppression
  1.0  perfect separation -- every retain score beats every forget score

Higher is better. Values below 0.5 mean the forget set scores *higher* than the
retain set, which is the opposite of what unlearning should produce.

Example
-------
python evaluation/auc_eval.py \
    --retain_csv outputs/15_forget_tgu/eval/ssim_retain/speaker_similarity_results.csv \
    --forget_csv outputs/15_forget_tgu/eval/ssim_forget/speaker_similarity_results.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="AUC between retain and forget speaker-similarity distributions"
    )
    parser.add_argument(
        "--retain_csv", required=True, help="ssim_eval.py output for the retain set"
    )
    parser.add_argument(
        "--forget_csv", required=True, help="ssim_eval.py output for the forget set"
    )
    parser.add_argument(
        "--column",
        default="cosine_similarity",
        help="Score column to read from both CSVs (default: cosine_similarity)",
    )
    parser.add_argument(
        "--output_file", default="", help="Optional JSON file to write the AUC to"
    )
    return parser.parse_args()


def read_scores(path: str, column: str) -> np.ndarray:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise SystemExit(f"{path} has no '{column}' column (found: {list(df.columns)})")
    scores = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
    if scores.size == 0:
        raise SystemExit(f"{path} contains no usable values in '{column}'")
    return scores


def main():
    args = parse_args()

    retain_scores = read_scores(args.retain_csv, args.column)
    forget_scores = read_scores(args.forget_csv, args.column)

    # Retain is the positive class: AUC = P(retain score > forget score).
    labels = np.concatenate(
        [np.ones(len(retain_scores)), np.zeros(len(forget_scores))]
    )
    scores = np.concatenate([retain_scores, forget_scores])
    auc = float(roc_auc_score(labels, scores))

    print(f"{auc:.6f}")

    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "auc": auc,
                    "retain_csv": args.retain_csv,
                    "forget_csv": args.forget_csv,
                    "column": args.column,
                    "retain_samples": int(retain_scores.size),
                    "forget_samples": int(forget_scores.size),
                },
                file_obj,
                indent=2,
            )


if __name__ == "__main__":
    main()
