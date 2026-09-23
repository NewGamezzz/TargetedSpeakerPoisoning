"""Shared helpers for the evaluation suite.

Every metric script consumes the same inference manifest (the CSVs shipped with
the HuggingFace metadata) and the output directory written by ``infer.py``.

Manifest columns
----------------
speaker_files   required  path to the reference / prompt wav
transcript      required  reference text used for synthesis
speaker_ids     optional  speaker id; derived from ``speaker_files`` when absent
output_files    optional  generated wav name; derived from ``speaker_files`` when absent
"""

import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

WAVLM_SV_MODEL = "microsoft/wavlm-base-plus-sv"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SR = 16000

# WavLM-TDNN downsamples by ~320x; anything shorter than a second makes the
# statistics-pooling layer degenerate, so short clips are zero-padded.
MIN_SAMPLES = TARGET_SR


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def speaker_id_from_path(speaker_file: str) -> str:
    """Derive a LibriTTS speaker id from ``.../<speaker>/<chapter>/<utt>.wav``."""
    parts = Path(speaker_file).parts
    if len(parts) >= 3:
        candidate = parts[-3]
        if candidate.isdigit():
            return candidate
    return Path(speaker_file).name.split("_")[0]


def load_manifest(inference_csv: str, root_path: str = "") -> pd.DataFrame:
    """Read an inference CSV and fill in the optional columns."""
    df = pd.read_csv(inference_csv)

    for column in ("speaker_files", "transcript"):
        if column not in df.columns:
            raise ValueError(f"{inference_csv} is missing the '{column}' column")

    df["reference_path"] = df["speaker_files"].apply(
        lambda p: os.path.join(root_path, p) if root_path else p
    )

    if "speaker_ids" in df.columns:
        df["speaker_id"] = df["speaker_ids"].astype(str)
    else:
        df["speaker_id"] = df["speaker_files"].apply(speaker_id_from_path)

    if "output_files" not in df.columns:
        df["output_files"] = df["speaker_files"].apply(
            lambda p: f"{Path(p).stem}.wav"
        )

    return df


def resolve_generated(gen_dir: str, row) -> list:
    """Return every generated wav belonging to one manifest row.

    Handles both layouts this project produces:

    * ``infer.py``  -> ``<gen_dir>/<reference stem>/sample_*.wav`` (one file per
      diffusion sample)
    * flat          -> ``<gen_dir>/<output_files>``, including the ``.wav.wav``
      names left by the original synthesis scripts
    """
    gen_dir = Path(gen_dir)
    output_name = Path(str(row["output_files"])).name
    ref_stem = Path(str(row["speaker_files"])).stem

    nested = gen_dir / ref_stem
    if nested.is_dir():
        samples = sorted(nested.glob("*.wav"))
        if samples:
            return samples

    for candidate in (
        gen_dir / output_name,
        gen_dir / f"{output_name}.wav",
        gen_dir / f"{Path(output_name).stem}.wav",
    ):
        if candidate.is_file():
            return [candidate]

    # Flat layout with several diffusion samples: ``<stem>_sample<N>.wav``.
    samples = sorted(gen_dir.glob(f"{Path(output_name).stem}_sample*.wav"))
    if samples:
        return samples

    return []


def iter_generated(df: pd.DataFrame, gen_dir: str):
    """Yield ``(row, generated_path)`` pairs, reporting rows with no audio."""
    missing = 0
    for _, row in df.iterrows():
        paths = resolve_generated(gen_dir, row)
        if not paths:
            missing += 1
            continue
        for path in paths:
            yield row, path
    if missing:
        print(f"Warning: {missing} manifest rows had no generated audio under {gen_dir}")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def load_audio(path, target_sr: int = TARGET_SR, min_samples: int = MIN_SAMPLES):
    """Load a wav as mono float at ``target_sr``, zero-padded to ``min_samples``."""
    audio, sr = sf.read(str(path))
    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if min_samples and len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    return np.asarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# Speaker embeddings
# ---------------------------------------------------------------------------
class SpeakerEncoder:
    """WavLM-TDNN (default) or ECAPA-TDNN speaker embeddings, L2-normalised."""

    def __init__(self, model_type: str = "wavlm", model_name: str = None, device=None):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            model_name = WAVLM_SV_MODEL if model_type == "wavlm" else ECAPA_MODEL
        self.model_name = model_name

        if model_type == "wavlm":
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.encoder = WavLMForXVector.from_pretrained(model_name).to(self.device)
            self.encoder.eval()
        else:
            from speechbrain.inference.speaker import EncoderClassifier

            self.feature_extractor = None
            self.encoder = EncoderClassifier.from_hparams(
                source=model_name, run_opts={"device": self.device}
            )
        print(f"Loaded speaker encoder {model_name} ({model_type}) on {self.device}")

    def embed(self, path) -> np.ndarray:
        audio = load_audio(path)

        if self.model_type == "wavlm":
            inputs = self.feature_extractor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.encoder(**inputs).embeddings
                emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb.squeeze().cpu().numpy()

        tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(tensor)
        return emb.squeeze().cpu().numpy()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two (possibly unnormalised) embeddings."""
    emb1 = np.asarray(emb1).flatten()
    emb2 = np.asarray(emb2).flatten()
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if denom == 0:
        return float("nan")
    return float(np.dot(emb1, emb2) / denom)
