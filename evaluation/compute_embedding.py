"""Pre-compute speaker embeddings for the enrolment set used by FSSIM.

Reads the ``speaker_files`` column of an inference CSV, embeds every reference
utterance with WavLM-TDNN and writes a ``.npy`` dictionary.

By default the dictionary is keyed by audio path (one entry per utterance);
``--average`` instead writes one centroid per speaker, which is what
``fssim_eval.py`` ultimately compares against. Either form is accepted there.

Example
-------
python evaluation/compute_embedding.py \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --root_path     /path/to/LibriTTS \
    --output_path   outputs/15_forget_tgu/forget_speaker_embeddings.npy
"""

import argparse
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from common import SpeakerEncoder, load_manifest, speaker_id_from_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute WavLM speaker embeddings and save them as .npy"
    )
    parser.add_argument("--inference_csv", required=True, help="CSV with a 'speaker_files' column")
    parser.add_argument("--root_path", default="", help="Root directory for LibriTTS wavs")
    parser.add_argument("--output_path", required=True, help="Destination .npy file")
    parser.add_argument(
        "--average",
        action="store_true",
        help="Save one centroid per speaker instead of one embedding per utterance",
    )
    parser.add_argument("--model_type", default="wavlm", choices=["wavlm", "ecapa"])
    parser.add_argument("--model_name", default=None, help="Override the encoder checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = load_manifest(args.inference_csv, args.root_path)

    # One reference utterance may be reused by several transcripts.
    references = df[["reference_path", "speaker_id"]].drop_duplicates("reference_path")
    print(f"Found {len(references)} unique reference utterances in {args.inference_csv}")

    encoder = SpeakerEncoder(model_type=args.model_type, model_name=args.model_name)

    embeddings = {}
    per_speaker = defaultdict(list)
    missing = 0

    for _, row in tqdm(references.iterrows(), total=len(references), desc="Computing embeddings"):
        path = row["reference_path"]
        if not os.path.exists(path):
            missing += 1
            continue
        emb = encoder.embed(path)
        embeddings[path] = emb
        per_speaker[str(row["speaker_id"])].append(emb)

    if missing:
        print(f"Warning: {missing} reference files were not found and were skipped")

    if args.average:
        payload = {
            speaker: np.mean(embs, axis=0) for speaker, embs in per_speaker.items()
        }
        np.save(args.output_path, payload)
        print(f"Saved {len(payload)} speaker centroids to {args.output_path}")
    else:
        np.save(args.output_path, embeddings)
        print(f"Saved {len(embeddings)} utterance embeddings to {args.output_path}")
        print(f"Covering {len(per_speaker)} speakers")


if __name__ == "__main__":
    main()
