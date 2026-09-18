# StyleTTS2 Speaker Unlearning

Machine unlearning for [StyleTTS2](https://github.com/yl4579/StyleTTS2) — fine-tunes the diffusion module to erase a target speaker's voice while preserving synthesis quality for all other speakers.


## Install
We test the code with Python 3.10.

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
**2. Install espeak**
```bash
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
git checkout 1.52.0
sudo apt-get install make autoconf automake libtool pkg-config libpcaudio-dev
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
```

Without root, configure with a user prefix (`./configure --prefix=$HOME/.local`)
and point phonemizer at the result, otherwise it raises
`RuntimeError: espeak not installed on your system`:

```bash
export PHONEMIZER_ESPEAK_LIBRARY=$HOME/.local/lib/libespeak-ng.so.1
export ESPEAK_DATA_PATH=$HOME/.local/share/espeak-ng-data
```

## Inference
**1. Download a model**
```bash
python download_weights.py --output_dir Models --setting 15 --mode tgu
```

**2. Download metadata**
```bash
python download_metadata.py --output_dir metadata --setting 15_forget_speakers --no_style_vectors
```

**3. Run inference**
Point `--model_dir` at any downloaded model folder. Each folder contains `config.yml` and `last.pth`.
```bash
python infer.py \
    --model_dir     Models/15_forget_tgu \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/15_forget_tgu
```

Generated audio will be in `outputs/15_forget_tgu/gen_files/`.


| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | required | Directory containing `config.yml` and `last.pth` |
| `--inference_csv` | required | CSV with `transcript` and `speaker_files` columns |
| `--root_path` | `""` | Root directory for LibriTTS `.wav` files |
| `--output_dir` | `./outputs` | Directory for generated audio |
| `--seed` | `0` | Random seed for utterance sampling |


## Evaluation

Six metrics, matching the paper. WER and UTMOS are reported for the retain and
the forget set alike; SSIM is reported for both and contrasted via AUC; FSSIM is
scored on the forget set.

| Metric | Script | Model | Good |
|--------|--------|-------|------|
| **WER** | `evaluation/wer_eval.py` | Whisper-medium | lower |
| **UTMOS** | `evaluation/mos_eval.py` | UTMOS | higher |
| **SSIM** | `evaluation/ssim_eval.py` | WavLM-TDNN (`microsoft/wavlm-base-plus-sv`) | higher on retain, lower on forget |
| **AUC** | `evaluation/auc_eval.py` | — | higher |
| **Avg-FSSIM** | `evaluation/fssim_eval.py` | WavLM-TDNN | lower |
| **Max-FSSIM** | `evaluation/fssim_eval.py` | WavLM-TDNN | lower |

**SSIM** is the easy condition: cosine similarity between a generated utterance
and the prompt it was conditioned on. **AUC** measures how separable the retain
and forget SSIM distributions are — 0.5 means they overlap completely, 1.0 means
perfect separation. **FSSIM** is the strong condition: it compares each generated
utterance against *every* speaker in the forget set, so a model that dodges the
prompt but still lands on another forgotten voice is caught. `Avg-FSSIM` averages
over those speakers and `Max-FSSIM` takes the worst case.

### How WER is computed

`wer_eval.py` reports the **mean of the per-utterance rates**, and excludes three
categories before averaging:

| Exclusion | Flag | Default |
|---|---|---|
| Utterance longer than 30s | `--max_duration_sec` | `30.0` |
| Reference shorter than 3 words | `--min_ref_words` | `3` |
| Hypothesis over 6 words/sec of audio | `--max_words_per_sec` | `6.0` |

Both of the latter two matter, and they matter in different situations. Against a
two-word reference a single substitution is already a 100% WER, so short
references dominate an unweighted mean while saying little about the model — on
the 15-speaker seen setting they move the reported figure by about 1.6 points. The
speaking-rate rule catches synthesis that has collapsed into babble; it fires on
no utterances at all for a healthy model, but removes ten points of WER from a
forget set whose output has degenerated.

Pass `0` to either flag to disable it. The summary JSON always records
`average_wer_unfiltered` alongside the headline figure, plus how many utterances
each rule removed, so the effect of the choice stays visible. It also records
`corpus_wer` — every edit pooled over every reference word, the usual ASR
convention — which runs roughly 0.5 points below the per-utterance mean here.

### Run everything

```bash
bash evaluation/run_eval.sh \
    Models/15_forget_tgu \
    metadata/LibriTTS/15_forget_speakers \
    /path/to/LibriTTS \
    outputs/15_forget_tgu
```

This runs inference on both test sets and writes every metric to
`outputs/15_forget_tgu/eval/`.

### Run one metric at a time

All scripts read the same inference CSV (`speaker_files`, `transcript`) and the
`gen_files/` directory written by `infer.py`.

```bash
# WER — run once per subset
python evaluation/wer_eval.py \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/retain_speaker_test_test_clean.csv \
    --gen_dir       outputs/15_forget_tgu/retain/gen_files \
    --output_dir    outputs/15_forget_tgu/eval/wer_retain

# UTMOS — run once per subset
python evaluation/mos_eval.py \
    --gen_dir     outputs/15_forget_tgu/retain/gen_files \
    --output_file outputs/15_forget_tgu/eval/utmos_retain.txt

# SSIM — run once per subset
python evaluation/ssim_eval.py \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --gen_dir       outputs/15_forget_tgu/forget/gen_files \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/15_forget_tgu/eval/ssim_forget

# AUC — consumes the two SSIM result CSVs
python evaluation/auc_eval.py \
    --retain_csv outputs/15_forget_tgu/eval/ssim_retain/speaker_similarity_results.csv \
    --forget_csv outputs/15_forget_tgu/eval/ssim_forget/speaker_similarity_results.csv

# FSSIM — enrol the forget speakers first, then score
python evaluation/compute_embedding.py \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --root_path     /path/to/LibriTTS \
    --output_path   outputs/15_forget_tgu/eval/forget_speaker_embeddings.npy

python evaluation/fssim_eval.py \
    --inference_csv   metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --gen_dir         outputs/15_forget_tgu/forget/gen_files \
    --embeddings_file outputs/15_forget_tgu/eval/forget_speaker_embeddings.npy \
    --output_file     outputs/15_forget_tgu/eval/fssim.csv
```

Every script writes a per-utterance CSV plus a `*_summary.json` holding the
aggregate. `auc_eval.py` prints the AUC alone and takes an optional
`--output_file`.

### Speaker filtering baseline

The paper compares against rejecting a prompt outright when it matches a forget
speaker. The threshold is **0.86**, the value the `wavlm-base-plus-sv` model card
gives for its verification decision.

### Reproduction

Run end to end against the published checkpoints on the full 4837-utterance test
sets, with no tuning. Paper figures are from
[arXiv:2603.07551](https://arxiv.org/abs/2603.07551).

**Table IV, 15 seen speakers, TGP** (`--setting 15 --mode tgu`):

| Metric | Ours ℛ | Ours ℱ | Paper ℛ | Paper ℱ |
|---|---|---|---|---|
| WER ↓ | 2.73 | 2.73 | 2.77 | 2.74 |
| MOS ↑ | 4.35 | 4.34 | 4.34 | 4.34 |
| SSIM (ℛ↑ ℱ↓) | 0.80 | 0.72 | 0.81 | 0.72 |
| AUC ↑ | 0.67 | — | 0.66 | — |
| Avg-FSSIM ↓ | — | 0.71 | — | 0.71 |
| Max-FSSIM ↓ | — | 0.91 | — | 0.91 |

**Table I, 15 unseen speakers, EGP+Triplet** (WER only):

| Metric | Ours ℛ | Ours ℱ | Paper ℛ | Paper ℱ |
|---|---|---|---|---|
| WER ↓ | 2.99 | 17.57 | 3.01 | 18.17 |

Every speaker metric lands within 0.01 of the paper. The second table is the
useful check on WER: most published WERs sit between 2.6 and 3.0, where a match
proves little, whereas EGP+Triplet degrades its forget set by roughly 6x and that
blow-up reproduces too.

Inference is deterministic — `infer.py` seeds torch at import, so a rerun of the
same manifest is bit-identical. Different subsets draw different diffusion noise,
which moves aggregate SSIM by about 0.003 and WER by about 0.2 points; per
utterance the same change moves SSIM by 0.12 on average, but the swings cancel.
WER is the one metric that is unstable on a subsample: even after filtering it
varies by 1.7 points across 500-utterance draws, because the distribution stays
heavy-tailed. Report it on the full test set.


<details>
<summary><h2>Pre-trained Models</h2></summary>

All checkpoints are at [NewGame/targeted-speaker-poisoning](https://huggingface.co/NewGame/targeted-speaker-poisoning).

```bash
# Download one model 
python download_weights.py --output_dir Models --setting 15 --mode tgu

# Download all models 
python download_weights.py --output_dir Models
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `Models` | Local directory to save weights |
| `--pretrained_only` | off | Download only the base pretrained checkpoint |
| `--setting` | all | `1`, `15`, or `100` |
| `--mode` | all | `tgu`, `tgu_triplet`, `egu`, or `egu_triplet` |

</details>


<details>
<summary><h2>Data</h2></summary>

Metadata for all three settings is on HuggingFace. The raw audio must be downloaded separately from [LibriTTS-R](https://www.openslr.org/141/).

```bash
# Download one setting (CSVs only, no style vectors)
python download_metadata.py --output_dir metadata --setting 15_forget_speakers --no_style_vectors

# Download one setting including style vectors 
python download_metadata.py --output_dir metadata --setting 15_forget_speakers

# Download all settings
python download_metadata.py --output_dir metadata
```

| Setting | Forget speakers | Dataset |
|---------|----------------|---------|
| `1_forget_speakers` | 1 | [NewGame/libritts-1-forget-speaker](https://huggingface.co/datasets/NewGame/libritts-1-forget-speaker) |
| `15_forget_speakers` | 15 | [NewGame/libritts-15-forget-speaker](https://huggingface.co/datasets/NewGame/libritts-15-forget-speaker) |
| `100_forget_speakers` | 100 | [NewGame/libritts-100-forget-speaker](https://huggingface.co/datasets/NewGame/libritts-100-forget-speaker) |

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `metadata` | Local directory to extract files into |
| `--setting` | all | `1_forget_speakers`, `15_forget_speakers`, or `100_forget_speakers` |
| `--no_style_vectors` | off | Skip downloading `style_vectors.zip` |

</details>


<details>
<summary><h2>Training</h2></summary>

Edit `data_params` in [Configs/config_unlearning.yml](Configs/config_unlearning.yml) to set your paths:

```yaml
data_params:
  dataset_path:        "metadata/LibriTTS/1_forget_speakers/train/diffusion_data.csv"
  forget_speaker_file: "metadata/LibriTTS/1_forget_speakers/train/forget_speaker_train.csv"
  root_path:           "/path/to/LibriTTS"
  style_root_path:     "metadata/LibriTTS"
```

**Standard mode (TGU / EGU):**
```bash
python train.py --config Configs/config_unlearning.yml --mode standard
```

**Triplet mode:**
```bash
python train.py --config Configs/config_unlearning.yml --mode triplet --lambda_triplet 1.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `Configs/config_unlearning.yml` | Path to YAML config |
| `--mode` | `standard` | `standard` or `triplet` |
| `--forget_ratio` | from config | Probability of substituting a forget-speaker reference [0, 1] |
| `--lambda_triplet` | from config | Triplet loss weight (triplet mode only) |
| `--max_iter` | `60,000` | Total gradient steps (cycles the dataloader; overrides epoch count) |
| `--comment` | None | Suffix appended to the run name |

Checkpoints are saved to `log_dir/` as `epoch_2nd_NNNNN.pth` and `last.pth`. TensorBoard logs are written to `log_dir/<run_name>/tensorboard/`. Launch the viewer with:

```bash
tensorboard --logdir Models/Unlearning
```

Set `tensorboard.enable: false` in the config to disable logging.

</details>


<details>
<summary><h2>Generating Style Vectors (optional)</h2></summary>

Style vectors are included in each HuggingFace dataset. To generate them yourself:

```bash
python gen_diffusion_ground_truth.py \
    --input_csv       metadata/LibriTTS/1_forget_speakers/train/retain_speaker_train.csv \
    --config_path     Models/pretrained/config.yml \
    --checkpoint_path Models/pretrained/epochs_2nd_00020.pth \
    --root_path       /path/to/LibriTTS \
    --output_dir      style_vectors \
    --diffusion_samples 2 \
    --resume
```

Outputs `style_vectors/diffusion/`, `style_vectors/ref/`, `diffusion_data.csv`, and `ref_data.csv`.

</details>


## Acknowledgements

This project builds directly on [StyleTTS2](https://github.com/yl4579/StyleTTS2) by Yinghao Aaron Li et al. Please cite the original work if you use this codebase.
