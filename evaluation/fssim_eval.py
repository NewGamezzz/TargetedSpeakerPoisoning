"""Forget Set Similarity (FSSIM) -- the paper's "strong condition".

SSIM only asks whether a generated utterance resembles the prompt it was given.
FSSIM is stricter: it compares every generated utterance against *every speaker
in the forget set*, so a model that dodges the prompt but still lands on some
other forgotten voice is caught.

Each forget speaker is enrolled as the centroid of their reference embeddings
(built by ``compute_embedding.py``), and two aggregates are reported:

  Avg-FSSIM  mean similarity over the forget speakers   (lower is better)
  Max-FSSIM  worst case over the forget speakers        (lower is better)

Following the original implementation, the utterance's own speaker is excluded
from both aggregates and reported separately as ``own_speaker_similarity``; pass
``--include_own`` to fold it in instead.

Example
-------
python evaluation/fssim_eval.py \
    --inference_csv   metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --gen_dir         outputs/15_forget_tgu/gen_files \
    --embeddings_file outputs/15_forget_tgu/forget_speaker_embeddings.npy \
    --output_file     outputs/15_forget_tgu/eval/fssim.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import SpeakerEncoder, cosine_similarity, load_manifest, resolve_generated, speaker_id_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="Forget Set Similarity (Avg-FSSIM / Max-FSSIM)")
    parser.add_argument("--inference_csv", required=True, help="CSV with 'speaker_files'")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated audio (gen_files)")
    parser.add_argument(
        "--embeddings_file",
        required=True,
        help="Forget-set embeddings .npy from compute_embedding.py",
    )
    parser.add_argument("--output_file", required=True, help="Destination CSV for per-utterance scores")
    parser.add_argument(
        "--include_own",
        action="store_true",
        help="Include the prompted speaker in the Avg/Max aggregates",
    )
    parser.add_argument("--model_type", default="wavlm", choices=["wavlm", "ecapa"])
    parser.add_argument("--model_name", default=None, help="Override the encoder checkpoint")
    return parser.parse_args()


def load_speaker_centroids(embeddings_file: str) -> dict:
    """Build ``{speaker_id: centroid}`` from either .npy layout.

    ``compute_embedding.py`` writes one embedding per utterance keyed by audio
    path; ``--average`` writes one centroid per speaker keyed by speaker id.
    Both are accepted here so existing enrolment files keep working.
    """
    data = np.load(embeddings_file, allow_pickle=True).item()
    if not data:
        raise SystemExit(f"{embeddings_file} contains no embeddings")

    sample_key = next(iter(data))
    already_averaged = "/" not in str(sample_key) and not str(sample_key).endswith(".wav")

    if already_averaged:
        centroids = {str(k): np.asarray(v) for k, v in data.items()}
        print(f"Loaded {len(centroids)} speaker centroids from {embeddings_file}")
        return centroids

    grouped = defaultdict(list)
    for audio_path, embedding in data.items():
        grouped[speaker_id_from_path(str(audio_path))].append(np.asarray(embedding))

    centroids = {speaker: np.mean(embs, axis=0) for speaker, embs in grouped.items()}
    print(
        f"Loaded {len(data)} utterance embeddings from {embeddings_file}, "
        f"averaged into {len(centroids)} speaker centroids"
    )
    return centroids


def main():
    args = parse_args()

    centroids = load_speaker_centroids(args.embeddings_file)
    forget_speakers = sorted(centroids)

    df = load_manifest(args.inference_csv)
    encoder = SpeakerEncoder(model_type=args.model_type, model_name=args.model_name)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing FSSIM"):
        speaker_id = str(row["speaker_id"])

        for generated_path in resolve_generated(args.gen_dir, row):
            try:
                embedding = encoder.embed(generated_path)
            except Exception as exc:
                print(f"Failed to embed {generated_path}: {exc}")
                continue

            similarities = {
                speaker: cosine_similarity(embedding, centroids[speaker])
                for speaker in forget_speakers
            }

            own_similarity = similarities.get(speaker_id, float("nan"))
            if args.include_own:
                aggregated = list(similarities.values())
            else:
                aggregated = [
                    value for speaker, value in similarities.items() if speaker != speaker_id
                ]

            results.append(
                {
                    "filename": generated_path.stem,
                    "speaker_id": speaker_id,
                    "generated_audio": str(generated_path),
                    "own_speaker_similarity": own_similarity,
                    "avg_fssim": float(np.mean(aggregated)) if aggregated else float("nan"),
                    "max_fssim": float(np.max(aggregated)) if aggregated else float("nan"),
                    **{f"similarity_to_{speaker}": value for speaker, value in similarities.items()},
                }
            )

    if not results:
        print("No generated audio was scored.")
        return

    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    summary = {
        "inference_csv": args.inference_csv,
        "gen_dir": args.gen_dir,
        "embeddings_file": args.embeddings_file,
        "forget_speakers": len(forget_speakers),
        "include_own": bool(args.include_own),
        "utterances": len(results_df),
        "avg_fssim": float(results_df["avg_fssim"].mean()),
        "max_fssim": float(results_df["max_fssim"].mean()),
        "own_speaker_similarity": float(results_df["own_speaker_similarity"].mean(skipna=True)),
    }

    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("\n" + "=" * 50)
    print("Forget Set Similarity complete")
    print(f"Utterances scored:  {summary['utterances']}")
    print(f"Forget speakers:    {summary['forget_speakers']}")
    print(f"Avg-FSSIM:          {summary['avg_fssim']:.4f}")
    print(f"Max-FSSIM:          {summary['max_fssim']:.4f}")
    print(f"Own-speaker SSIM:   {summary['own_speaker_similarity']:.4f}")
    print(f"Results CSV: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
