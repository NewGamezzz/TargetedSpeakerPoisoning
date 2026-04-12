import torch
import torchaudio
import librosa
import numpy as np

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave, to_mel=None):
    # Initialize to_mel if not provided
    if to_mel is None:
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4

    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, model, device="cuda"):
    wave, sr = librosa.load(path, sr=24000)
    # print(wave.shape)
    audio, index = librosa.effects.trim(wave, top_db=30)
    # print(audio.shape)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    # padding to 3 seconds if shorter
    if audio.shape[0] < 1 * 24000:
        audio = np.pad(audio, (0, 1 * 24000 - audio.shape[0]), 'constant', constant_values=(0, 0))
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)