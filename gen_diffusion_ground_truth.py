"""
Pre-compute style vectors for the StyleTTS2 unlearning training pipeline.

For each utterance in an input CSV this script runs the StyleTTS2 diffusion
sampler and saves two style tensors per sample:
  - diffusion vector  (.pt)  — s_pred from the diffusion model
  - reference vector  (.pt)  — s_ref from the style/predictor encoders

It also writes diffusion_data.csv and ref_data.csv (with relative paths) that
can be passed directly to train.py via dataset_path in the config.

Usage
-----
  python gen_diffusion_ground_truth.py \\
      --input_csv      metadata/retain_speaker_train.csv \\
      --config_path    Configs/config_unlearning.yml \\
      --checkpoint_path Models/LibriTTS/epochs_2nd_00020.pth \\
      --output_dir     style_vectors \\
      --diffusion_samples 2 \\
      --resume

Input CSV columns
-----------------
  speaker_files   : path to the reference .wav (resolved against --root_path)
  transcript      : raw English text
  speaker_ids     : speaker identifier
  output_files    : base filename (no extension) for the output .pt files

Output
------
  <output_dir>/diffusion/<base>_diffusion{i}.pt
  <output_dir>/ref/<base>_ref.pt
  <output_dir>/diffusion_data.csv
  <output_dir>/ref_data.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import phonemizer
import torch
import torchaudio
import yaml
from munch import Munch
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from models import *
from utils import *
from Utils.inference import compute_style
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from text_utils import TextCleaner

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_cleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(config_path: str, checkpoint_path: str):
    config = yaml.safe_load(open(config_path))

    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval().to(device) for key in model]

    params_whole = torch.load(checkpoint_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key not in params:
            continue
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            from collections import OrderedDict
            new_sd = OrderedDict(
                (k[7:] if k.startswith("module.") else k, v)
                for k, v in params[key].items()
            )
            model[key].load_state_dict(new_sd, strict=False)
        print(f"  Loaded: {key}")

    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
    return model, sampler, model_params


# ---------------------------------------------------------------------------
# Style vector computation
# ---------------------------------------------------------------------------
def compute_diffusion_style(
    model,
    sampler,
    text: str,
    ref_s,
    diffusion_steps: int = 5,
):
    """Return (pred_s, ref_s) style tensors for a single utterance."""
    ps = global_phonemizer.phonemize([text])
    ps = " ".join(word_tokenize(ps[0]))
    tokens = text_cleaner(ps)
    tokens = torch.LongTensor([0] + tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())

        pred_s = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=1,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

    return pred_s, ref_s


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------
def _save_tensor(tensor, path):
    tmp = f"{path}.part"
    torch.save(tensor, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(os.path.join(args.output_dir, "diffusion"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ref"), exist_ok=True)

    model, sampler, _ = load_model(args.config_path, args.checkpoint_path)

    df = pd.read_csv(args.input_csv)

    diffusion_rows = []
    ref_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating style vectors"):
        text = str(row["transcript"]).strip()
        ref_path = os.path.join(args.root_path, str(row["speaker_files"]))
        speaker_id = row["speaker_ids"]
        output_base = row["output_files"]

        ps = " ".join(word_tokenize(global_phonemizer.phonemize([text])[0]))

        # Reference style is shared across all diffusion samples for this row
        ref_out_name = f"{output_base}_ref.pt"
        ref_out_path = os.path.join(args.output_dir, "ref", ref_out_name)
        ref_rel_path = os.path.join("ref", ref_out_name)

        try:
            ref_s = compute_style(ref_path, model, device)
        except Exception as e:
            print(f"Skipping {ref_path}: {e}")
            continue

        if not (args.resume and os.path.exists(ref_out_path)):
            _save_tensor(ref_s.cpu(), ref_out_path)

        ref_rows.append({
            "filepath": ref_rel_path,
            "text": ps,
            "speaker_id": speaker_id,
            "ref_filepath": str(row["speaker_files"]),
        })

        for i in range(args.diffusion_samples):
            diff_out_name = f"{output_base}_diffusion{i}.pt"
            diff_out_path = os.path.join(args.output_dir, "diffusion", diff_out_name)
            diff_rel_path = os.path.join("diffusion", diff_out_name)

            if args.resume and os.path.exists(diff_out_path):
                diffusion_rows.append({
                    "filepath": diff_rel_path,
                    "text": ps,
                    "speaker_id": speaker_id,
                    "ref_filepath": str(row["speaker_files"]),
                })
                continue

            try:
                pred_s, _ = compute_diffusion_style(
                    model, sampler, text, ref_s, args.diffusion_steps
                )
                _save_tensor(pred_s.cpu(), diff_out_path)
                diffusion_rows.append({
                    "filepath": diff_rel_path,
                    "text": ps,
                    "speaker_id": speaker_id,
                    "ref_filepath": str(row["speaker_files"]),
                })
            except Exception as e:
                print(f"Error on sample {i} for {output_base}: {e}")

    # Save metadata CSVs
    pd.DataFrame(diffusion_rows).to_csv(
        os.path.join(args.output_dir, "diffusion_data.csv"), index=False
    )
    ref_df = pd.DataFrame(ref_rows).drop_duplicates(
        subset=["filepath", "text", "speaker_id", "ref_filepath"]
    )
    ref_df.to_csv(os.path.join(args.output_dir, "ref_data.csv"), index=False)

    print(f"diffusion_data.csv: {len(diffusion_rows)} rows")
    print(f"ref_data.csv:       {len(ref_df)} rows")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute StyleTTS2 diffusion style vectors for unlearning training."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV with columns: speaker_files, transcript, speaker_ids, output_files.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="Configs/config_unlearning.yml",
        help="Path to the StyleTTS2 config YAML.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="Models/LibriTTS/epochs_2nd_00020.pth",
        help="Path to the pretrained StyleTTS2 checkpoint.",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root directory for LibriTTS .wav files (prepended to speaker_files paths).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="style_vectors",
        help="Directory to write style vectors and metadata CSVs.",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=2,
        help="Number of independently sampled diffusion vectors per utterance.",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=5,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip utterances whose output .pt files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
