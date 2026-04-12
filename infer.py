
import os
import shutil
import argparse
import random

import numpy as np
import torch
import torchaudio
import yaml
from munch import Munch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import phonemizer
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from Utils.inference import compute_style
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from text_utils import TextCleaner

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_cleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(config_path: str, checkpoint_path: str):
    """Load a trained unlearning checkpoint and return (model, sampler, model_params)."""
    config = yaml.safe_load(open(config_path))

    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[k].eval().to(device) for k in model]

    params_whole = torch.load(checkpoint_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key not in params:
            continue
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            # Strip DataParallel "module." prefix if present
            from collections import OrderedDict
            new_sd = OrderedDict(
                (k[7:] if k.startswith("module.") else k, v)
                for k, v in params[key].items()
            )
            model[key].load_state_dict(new_sd, strict=False)
        print(f"  Loaded: {key}")

    _ = [model[k].eval() for k in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
    return model, sampler, model_params


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def synthesise(
    model,
    sampler,
    model_params,
    text: str,
    ref_style,
    alpha: float = 1.0,
    beta: float = 1.0,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    is_raw_text: bool = True,
):
    """
    Generate a waveform for `text` conditioned on `ref_style`.

    Parameters
    ----------
    alpha : float
        Interpolation weight for the style-encoder component of the prediction.
        0 = keep original reference style; 1 = use fully predicted style.
    beta  : float
        Same for the predictor-encoder component.

    Returns
    -------
    wav : np.ndarray   — synthesised audio (24 kHz)
    s_pred : Tensor    — style prediction from the diffusion model
    """
    if is_raw_text:
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = " ".join(word_tokenize(ps[0]))
    else:
        ps = text

    tokens = text_cleaner(ps)
    tokens = torch.LongTensor([0] + tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_style,
            num_steps=diffusion_steps,
        ).squeeze(1)

        # Interpolate between diffusion prediction and reference
        s = alpha * s_pred[:, 128:] + (1 - alpha) * ref_style[:, 128:]
        ref = beta * s_pred[:, :128] + (1 - beta) * ref_style[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c = 0
        for i in range(pred_aln.size(0)):
            pred_aln[i, c : c + int(pred_dur[i].data)] = 1
            c += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, :-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, :-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    # trim trailing artifact
    return out.squeeze().cpu().numpy()[..., :-50], s_pred


# ---------------------------------------------------------------------------
# Main inference runner
# ---------------------------------------------------------------------------
def run_inference(args):
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Alpha={args.alpha}  Beta={args.beta}  Diffusion steps={args.diffusion_steps}")

    model, sampler, model_params = load_model(args.config, args.checkpoint)

    inference_df = pd.read_csv(args.inference_csv)
    if args.utterance_samples > 0:
        inference_df = inference_df.sample(
            n=min(args.utterance_samples, len(inference_df)),
            random_state=args.seed,
        )

    ref_dir = os.path.join(args.output_dir, "ref_files")
    gen_dir = os.path.join(args.output_dir, "gen_files")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    for idx, row in tqdm(inference_df.iterrows(), total=len(inference_df), desc="Inference"):
        text = row["transcript"]
        ref_path = row["speaker_files"]

        # Copy reference to output for easy side-by-side comparison
        shutil.copy(ref_path, os.path.join(ref_dir, os.path.basename(ref_path)))

        try:
            ref_style = compute_style(ref_path, model, device)
        except Exception as e:
            print(f"Failed to compute style for {ref_path}: {e}")
            continue

        spk_gen_dir = os.path.join(gen_dir, Path(ref_path).stem)
        os.makedirs(spk_gen_dir, exist_ok=True)

        for sample_idx in range(max(1, args.diffusion_samples)):
            try:
                wav, _ = synthesise(
                    model, sampler, model_params,
                    text, ref_style,
                    alpha=args.alpha,
                    beta=args.beta,
                    diffusion_steps=args.diffusion_steps,
                    is_raw_text=True,
                )
                out_path = os.path.join(spk_gen_dir, f"sample_{sample_idx}.wav")
                torchaudio.save(out_path, torch.from_numpy(wav).unsqueeze(0), 24000)
            except Exception as e:
                print(f"Error generating sample {sample_idx} for row {idx}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="StyleTTS2 unlearning inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML (typically copied into the run directory by train.py).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth). Use 'last.pth' or a specific epoch file.",
    )
    parser.add_argument(
        "--inference_csv",
        type=str,
        required=True,
        help="CSV with columns: transcript, speaker_files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save reference copies and generated waveforms.",
    )
    parser.add_argument(
        "--utterance_samples",
        type=int,
        default=0,
        help="Randomly sample this many utterances from inference_csv (0 = use all).",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=1,
        help="Number of independently sampled waveforms per utterance.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Style-encoder interpolation weight (0=reference, 1=predicted).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Predictor-encoder interpolation weight (0=reference, 1=predicted).",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=5,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling utterances.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(_parse_args())
