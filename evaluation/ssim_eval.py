"""Speaker similarity between a generated utterance and its own prompt (SSIM).

This is the "easy condition" of the paper: prompt-output similarity. The same
script serves both columns of the results tables -- run it once with a retain
manifest and once with a forget manifest.

  retain set (R)  higher is better -- the model still clones permitted voices
  forget set (F)  lower  is better -- the target voice has been suppressed

The two output CSVs are what ``auc_eval.py`` consumes.

Example
-------
python evaluation/ssim_eval.py \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --gen_dir       outputs/15_forget_tgu/gen_files \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/15_forget_tgu/eval/ssim_forget
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import SpeakerEncoder, cosine_similarity, load_manifest, resolve_generated


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt-output speaker similarity (SSIM)")
    parser.add_argument("--inference_csv", required=True, help="CSV with 'speaker_files'")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated audio (gen_files)")
    parser.add_argument("--root_path", default="", help="Root directory for LibriTTS wavs")
    parser.add_argument("--output_dir", required=True, help="Directory for the similarity results")
    parser.add_argument("--model_type", default="wavlm", choices=["wavlm", "ecapa"])
    parser.add_argument("--model_name", default=None, help="Override the encoder checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "speaker_similarity_results.csv"
    summary_json = output_dir / "speaker_similarity_summary.json"

    df = load_manifest(args.inference_csv, args.root_path)
    encoder = SpeakerEncoder(model_type=args.model_type, model_name=args.model_name)

    # One reference is often reused across transcripts; embed each one once.
    reference_cache = {}
    similarity_scores = []
    successful = failed = 0

    with open(results_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            ["filename", "cosine_similarity", "speaker_id", "generated_audio", "reference_audio"]
        )

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SSIM"):
            reference_path = row["reference_path"]
            if not os.path.exists(reference_path):
                print(f"Reference audio not found: {reference_path}")
                failed += 1
                continue

            generated_paths = resolve_generated(args.gen_dir, row)
            if not generated_paths:
                failed += 1
                continue

            if reference_path not in reference_cache:
                reference_cache[reference_path] = encoder.embed(reference_path)
            emb_reference = reference_cache[reference_path]

            for generated_path in generated_paths:
                try:
                    emb_generated = encoder.embed(generated_path)
                except Exception as exc:
                    print(f"Failed to embed {generated_path}: {exc}")
                    failed += 1
                    continue

                similarity = cosine_similarity(emb_generated, emb_reference)
                if np.isnan(similarity):
                    print(f"Skipping {generated_path.name}: degenerate embedding")
                    failed += 1
                    continue

                writer.writerow(
                    [
                        generated_path.stem,
                        similarity,
                        row["speaker_id"],
                        str(generated_path),
                        reference_path,
                    ]
                )
                similarity_scores.append(similarity)
                successful += 1

    if not similarity_scores:
        print("No files were successfully processed!")
        return

    summary = {
        "inference_csv": args.inference_csv,
        "gen_dir": args.gen_dir,
        "model_name": encoder.model_name,
        "successful": successful,
        "failed": failed,
        "average_similarity": float(np.mean(similarity_scores)),
        "min_similarity": float(np.min(similarity_scores)),
        "max_similarity": float(np.max(similarity_scores)),
    }

    with open(summary_json, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("\nSpeaker similarity evaluation completed")
    print(f"Successfully processed: {successful} files ({failed} failed)")
    print(f"Average similarity (SSIM): {summary['average_similarity']:.4f}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
