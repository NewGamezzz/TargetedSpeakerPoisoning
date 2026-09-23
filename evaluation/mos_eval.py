"""UTMOS naturalness score of the generated speech (utility metric).

UTMOS is an automatic mean-opinion-score predictor on the usual 1-5 scale; it
stands in for a listening test. Higher is better.

By default every wav under ``--gen_dir`` is scored. Pass ``--inference_csv`` to
restrict scoring to the files named by a manifest.

Example
-------
python evaluation/mos_eval.py \
    --gen_dir     outputs/15_forget_tgu/gen_files \
    --output_file outputs/15_forget_tgu/eval/utmos_scores.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import load_manifest, resolve_generated


def parse_args():
    parser = argparse.ArgumentParser(description="UTMOS inference on generated audio")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated audio (gen_files)")
    parser.add_argument("--output_file", required=True, help="Destination .txt (TSV) file")
    parser.add_argument(
        "--inference_csv",
        default="",
        help="Optional manifest; when given, only files it references are scored",
    )
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for directory mode")
    return parser.parse_args()


def load_utmos_model():
    try:
        import utmos
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "utmos is not installed in the active environment.\n"
            "Install it with:  pip install utmos==1.1.10"
        ) from exc

    print("Loading UTMOS model...")
    return utmos.Score()


def collect_files(args):
    gen_dir = Path(args.gen_dir).expanduser().resolve()
    if not gen_dir.is_dir():
        raise SystemExit(f"gen_dir does not exist or is not a directory: {gen_dir}")

    if args.inference_csv:
        df = load_manifest(args.inference_csv)
        paths = []
        for _, row in df.iterrows():
            paths.extend(resolve_generated(gen_dir, row))
        if not paths:
            raise SystemExit(f"No manifest row resolved to audio under {gen_dir}")
        return sorted(set(paths))

    # infer.py writes one sub-directory per utterance, so search recursively.
    paths = sorted(gen_dir.rglob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matching '{args.pattern}' found under {gen_dir}")
    return paths


def main():
    args = parse_args()

    wav_files = collect_files(args)
    print(f"Found {len(wav_files)} audio files to process")

    model = load_utmos_model()

    results = []
    for wav_file in tqdm(wav_files, desc="Scoring"):
        try:
            results.append((wav_file.name, float(model.calculate_wav_file(str(wav_file)))))
        except Exception as exc:
            print(f"  Error processing {wav_file.name}: {exc}")
            results.append((wav_file.name, "ERROR"))

    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("Filename\tUTMOS_Score\n")
        for filename, score in results:
            file_obj.write(f"{filename}\t{score}\n")

    print(f"\nResults saved to {output_path}")

    valid = [score for _, score in results if score != "ERROR"]
    if not valid:
        print("No files were scored successfully.")
        return

    summary = {
        "gen_dir": str(args.gen_dir),
        "total_files": len(results),
        "successful": len(valid),
        "errors": len(results) - len(valid),
        "average_utmos": float(np.mean(valid)),
        "min_utmos": float(np.min(valid)),
        "max_utmos": float(np.max(valid)),
    }

    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("Summary:")
    print(f"  Successful: {summary['successful']} / {summary['total_files']}")
    print(f"  Average UTMOS: {summary['average_utmos']:.4f}")
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
