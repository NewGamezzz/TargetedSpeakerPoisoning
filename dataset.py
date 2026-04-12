# coding: utf-8
"""
Unified dataset for speaker unlearning fine-tuning.

Supports two modes:
  - 'standard': returns (speaker_id, text, s_trg, ref_mel)
                The reference mel is randomly replaced with a forget-speaker
                sample at probability `forget_ratio`.
  - 'triplet' : returns (speaker_id, text, s_trg, ref_mel, neg_mel, forget_flag)
                When a forget sample is drawn, a second independent forget-speaker
                file is used as the triplet negative.
"""

import os
import os.path as osp
import random

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Text tokenisation
# ---------------------------------------------------------------------------
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋ"
    "ⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_char_to_id = {ch: i for i, ch in enumerate(symbols)}


class TextCleaner:
    def __call__(self, text: str):
        return [_char_to_id[ch] for ch in text if ch in _char_to_id]


# ---------------------------------------------------------------------------
# Mel-spectrogram helpers
# ---------------------------------------------------------------------------
np.random.seed(1)
random.seed(1)

_MEL_PARAMS = {"n_mels": 80, "n_fft": 2048, "win_length": 1200, "hop_length": 300}
_to_mel = torchaudio.transforms.MelSpectrogram(**_MEL_PARAMS)
_LOG_MEAN, _LOG_STD = -4.0, 4.0


def _preprocess_wave(wave: np.ndarray) -> torch.Tensor:
    """Convert a raw waveform array to a normalised log-mel spectrogram."""
    wave_t = torch.from_numpy(wave).float()
    mel = _to_mel(wave_t)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - _LOG_MEAN) / _LOG_STD
    return mel  # (1, n_mels, T)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FilePathDataset(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    data_list : pd.DataFrame
        Each row: (filepath, text, speaker_id, speaker_dir)
        where `filepath` points to a pre-computed style vector (.pt).
    mode : {'standard', 'triplet'}
        'standard' — baseline unlearning (reference replacement only).
        'triplet'  — unlearning with triplet contrastive loss.
    forget_speaker_file : list[str]
        Paths to audio files of the speaker to be forgotten.
    forget_ratio : float
        Probability of substituting the reference with a forget-speaker sample.
    """

    MAX_MEL_LENGTH = 192

    def __init__(
        self,
        data_list,
        root_path,
        style_root_path,
        sr: int = 24000,
        data_augmentation: bool = False,
        validation: bool = False,
        OOD_data: str = "Data/OOD_texts.txt",
        min_length: int = 50,
        forget_speaker_file=None,
        forget_ratio: float = 0.8,
        mode: str = "standard",
    ):
        assert mode in ("standard", "triplet"), (
            f"Unknown mode '{mode}'. Choose 'standard' or 'triplet'."
        )
        self.df = data_list
        self.sr = sr
        self.root_path = root_path          # root for LibriTTS .wav files
        self.style_root_path = style_root_path  # root for pre-computed style vectors (.pt)
        self.text_cleaner = TextCleaner()
        self.data_augmentation = data_augmentation and (not validation)
        self.min_length = min_length
        self.forget_speaker_file = [
            osp.join(root_path, f) for f in (forget_speaker_file or [])
        ]
        self.forget_ratio = forget_ratio
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        if self.mode == "triplet":
            return self._get_triplet(data)
        else:
            return self._get_standard(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_wav(self, filepath) -> np.ndarray:
        wav, sr = sf.read(str(filepath))
        if wav.ndim == 2:
            wav = wav[:, 0]
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        # pad with silence at both ends for stable mel computation
        wav = np.concatenate([np.zeros(5000), wav, np.zeros(5000)])
        return wav

    def _wav_to_mel(self, wav: np.ndarray) -> torch.Tensor:
        mel = _preprocess_wave(wav).squeeze(0)  # (n_mels, T)
        T = mel.size(1)
        if T > self.MAX_MEL_LENGTH:
            start = np.random.randint(0, T - self.MAX_MEL_LENGTH)
            mel = mel[:, start : start + self.MAX_MEL_LENGTH]
        return mel

    def _tokenise(self, text: str) -> torch.LongTensor:
        ids = self.text_cleaner(text)
        ids = [0] + ids + [0]
        return torch.LongTensor(ids)

    def _sample_forget_file(self) -> str:
        return self.forget_speaker_file[
            np.random.randint(0, len(self.forget_speaker_file))
        ]

    def _load_row(self, data):
        """Parse a dataframe row and return (s_trg, text_tensor, speaker_dir)."""
        filepath, text, speaker_id, speaker_dir = data
        filepath = osp.join(self.style_root_path, filepath)
        speaker_dir = osp.join(self.root_path, speaker_dir).rsplit("/", 2)[0]
        speaker_id = int(speaker_id)
        s_trg = torch.load(filepath)
        text_tensor = self._tokenise(text)
        return s_trg, text_tensor, speaker_id, speaker_dir

    # ------------------------------------------------------------------
    # Mode-specific item builders
    # ------------------------------------------------------------------
    def _get_standard(self, data):
        s_trg, text_tensor, speaker_id, speaker_dir = self._load_row(data)

        ref_files = list(Path(speaker_dir).rglob("*.wav"))
        ref_file = str(np.random.choice(ref_files))

        if np.random.rand() < self.forget_ratio:
            ref_file = self._sample_forget_file()

        ref_mel = self._wav_to_mel(self._read_wav(ref_file))
        return speaker_id, text_tensor, s_trg, ref_mel

    def _get_triplet(self, data):
        s_trg, text_tensor, speaker_id, speaker_dir = self._load_row(data)

        ref_files = list(Path(speaker_dir).rglob("*.wav"))
        ref_file = str(np.random.choice(ref_files))

        forget_sample = 0
        neg_file = ref_file  # default: same as ref (non-forget batch)

        if np.random.rand() < self.forget_ratio:
            ref_file = self._sample_forget_file()
            neg_file = self._sample_forget_file()  # independent second forget sample
            forget_sample = 1

        ref_mel = self._wav_to_mel(self._read_wav(ref_file))
        neg_mel = self._wav_to_mel(self._read_wav(neg_file))

        return speaker_id, text_tensor, s_trg, ref_mel, neg_mel, forget_sample


# ---------------------------------------------------------------------------
# Collater
# ---------------------------------------------------------------------------
class Collater:
    """Pads and stacks a batch returned by FilePathDataset."""

    def __init__(self, mode: str = "standard"):
        self.mode = mode

    @staticmethod
    def _pad_texts(texts):
        B = len(texts)
        max_len = max(t.size(0) for t in texts)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, t in enumerate(texts):
            n = t.size(0)
            padded[i, :n] = t
            lengths[i] = n
        return padded, lengths

    @staticmethod
    def _pad_mels(mels):
        B = len(mels)
        n_mels = mels[0].size(0)
        max_T = max(m.size(1) for m in mels)
        padded = torch.zeros(B, n_mels, max_T, dtype=torch.float)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, m in enumerate(mels):
            T = m.size(1)
            padded[i, :, :T] = m
            lengths[i] = T
        return padded, lengths

    def __call__(self, batch):
        if self.mode == "triplet":
            return self._collate_triplet(batch)
        return self._collate_standard(batch)

    def _collate_standard(self, batch):
        """
        Returns:
            texts            (B, L_text)
            input_lengths    (B,)
            s_trg            (B, style_dim)
            ref_mels         (B, n_mels, T_ref)
            ref_mels_length  (B,)
        """
        texts_raw = [b[1] for b in batch]
        s_trg = torch.stack([b[2] for b in batch]).squeeze(1)
        mels_raw = [b[3] for b in batch]

        texts, input_lengths = self._pad_texts(texts_raw)
        ref_mels, ref_mels_length = self._pad_mels(mels_raw)

        return texts, input_lengths, s_trg, ref_mels, ref_mels_length

    def _collate_triplet(self, batch):
        """
        Returns:
            texts                (B, L_text)
            input_lengths        (B,)
            s_trg                (B, style_dim)
            ref_mels             (B, n_mels, T_ref)
            ref_mels_length      (B,)
            negative_mels        (B, n_mels, T_neg)
            negative_mels_length (B,)
            forget_samples       (B,)   — 1 if forget speaker, else 0
        """
        texts_raw = [b[1] for b in batch]
        s_trg = torch.stack([b[2] for b in batch]).squeeze(1)
        ref_raw = [b[3] for b in batch]
        neg_raw = [b[4] for b in batch]
        forget_flags = [b[5] for b in batch]

        texts, input_lengths = self._pad_texts(texts_raw)
        ref_mels, ref_mels_length = self._pad_mels(ref_raw)
        neg_mels, neg_mels_length = self._pad_mels(neg_raw)

        return (
            texts,
            input_lengths,
            s_trg,
            ref_mels,
            ref_mels_length,
            neg_mels,
            neg_mels_length,
            torch.tensor(forget_flags, dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloader(
    path_list,
    root_path,
    style_root_path,
    mode: str = "standard",
    validation: bool = False,
    OOD_data: str = "Data/OOD_texts.txt",
    min_length: int = 50,
    batch_size: int = 4,
    num_workers: int = 1,
    device="cpu",
    dataset_config=None,
):
    """
    Parameters
    ----------
    path_list : pd.DataFrame or str
        Training data (DataFrame or path to CSV).
    root_path : str
        Root directory for LibriTTS .wav files.
    style_root_path : str
        Root directory for pre-computed style vectors (.pt files).
    mode : {'standard', 'triplet'}
        Which unlearning mode to use.
    dataset_config : dict
        Forwarded to FilePathDataset (e.g. forget_speaker_file, forget_ratio).
    """
    if dataset_config is None:
        dataset_config = {}

    if isinstance(path_list, str) and osp.exists(path_list):
        path_list = pd.read_csv(path_list)

    dataset = FilePathDataset(
        path_list,
        root_path,
        style_root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        validation=validation,
        mode=mode,
        **dataset_config,
    )
    collate_fn = Collater(mode=mode)

    if isinstance(device, str):
        pin = "cuda" in device and torch.cuda.is_available()
    else:
        pin = torch.cuda.is_available() and getattr(device, "type", "cpu") == "cuda"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not validation,
        num_workers=num_workers,
        drop_last=not validation,
        collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
