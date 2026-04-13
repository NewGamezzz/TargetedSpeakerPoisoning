# StyleTTS2 Speaker Unlearning

Machine unlearning for [StyleTTS2](https://github.com/yl4579/StyleTTS2) — fine-tunes the diffusion module to erase a target speaker's voice while preserving synthesis quality for all other speakers.

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download a model**
```bash
python download_weights.py --output_dir Models --setting 15 --mode tgu
```

**3. Download metadata**
```bash
python download_metadata.py --output_dir metadata --setting 15_forget_speakers --no_style_vectors
```

**4. Run inference**
```bash
python infer.py \
    --model_dir     Models/15_forget_tgu \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/15_forget_tgu
```

Generated audio will be in `outputs/15_forget_tgu/gen_files/`.

---

## Inference

Point `--model_dir` at any downloaded model folder. Each folder contains `config.yml` and `last.pth`.

```bash
python infer.py \
    --model_dir     Models/15_forget_tgu \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test.csv \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | required | Directory containing `config.yml` and `last.pth` |
| `--inference_csv` | required | CSV with `transcript` and `speaker_files` columns |
| `--root_path` | `""` | Root directory for LibriTTS `.wav` files |
| `--output_dir` | `./outputs` | Directory for generated audio |
| `--utterance_samples` | `0` (all) | Number of utterances to randomly sample (0 = use all) |
| `--diffusion_samples` | `1` | Independent waveforms to generate per utterance |
| `--alpha` | `1.0` | Style-encoder interpolation (0 = reference, 1 = predicted) |
| `--beta` | `1.0` | Predictor-encoder interpolation (0 = reference, 1 = predicted) |
| `--diffusion_steps` | `5` | Number of diffusion sampling steps |
| `--seed` | `0` | Random seed for utterance sampling |

---

## Evaluation

To assess unlearning effectiveness, run inference separately on the forget-speaker and retain-speaker test sets and compare the generated audio.

**Forget speakers** (voices the model should have forgotten):
```bash
python infer.py \
    --model_dir     Models/15_forget_tgu \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test.csv \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/eval_forget
```

**Retain speakers** (voices the model should preserve):
```bash
python infer.py \
    --model_dir     Models/15_forget_tgu \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/retain_speaker_test.csv \
    --root_path     /path/to/LibriTTS \
    --output_dir    outputs/eval_retain
```

Setting `--alpha 1.0 --beta 1.0` (the default) makes the model rely entirely on its diffusion-predicted style, which is the standard setup for unlearning assessment.

Use `forget_speaker_test_test_clean.csv` and `retain_speaker_test_test_clean.csv` for evaluation on the LibriTTS test-clean subset only.

---

## Pre-trained Models

All checkpoints are at [NewGame/targeted-speaker-poisoning](https://huggingface.co/NewGame/targeted-speaker-poisoning).

```bash
# Download one model (~969 MB)
python download_weights.py --output_dir Models --setting 15 --mode tgu

# Download all models (~12.4 GB)
python download_weights.py --output_dir Models
```

| Folder | Forget speakers | Method |
|--------|----------------|--------|
| `pretrained/` | — | Base StyleTTS2 |
| `1_forget_tgu/` | 1 | Standard |
| `1_forget_tgu_triplet/` | 1 | Triplet |
| `1_forget_egu/` | 1 | Standard (EGU) |
| `1_forget_egu_triplet/` | 1 | Triplet (EGU) |
| `15_forget_tgu/` | 15 | Standard |
| `15_forget_tgu_triplet/` | 15 | Triplet |
| `15_forget_egu/` | 15 | Standard (EGU) |
| `15_forget_egu_triplet/` | 15 | Triplet (EGU) |
| `100_forget_tgu/` | 100 | Standard |
| `100_forget_tgu_triplet/` | 100 | Triplet |
| `100_forget_egu/` | 100 | Standard (EGU) |
| `100_forget_egu_triplet/` | 100 | Triplet (EGU) |

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `Models` | Local directory to save weights |
| `--pretrained_only` | off | Download only the base pretrained checkpoint |
| `--setting` | all | `1`, `15`, or `100` |
| `--mode` | all | `tgu`, `tgu_triplet`, `egu`, or `egu_triplet` |
| `--token` | None | HuggingFace token (for private repos) |

---

## Data

Metadata for all three settings is on HuggingFace. The raw audio must be downloaded separately from [LibriTTS](https://www.openslr.org/60/).

```bash
# Download one setting (CSVs only, no style vectors)
python download_metadata.py --output_dir metadata --setting 15_forget_speakers --no_style_vectors

# Download one setting including style vectors (~2.3 GB)
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
| `--token` | None | HuggingFace token (for private repos) |

---

## Training

Edit `data_params` in [Configs/config_unlearning.yml](Configs/config_unlearning.yml) to set your paths:

```yaml
data_params:
  dataset_path:        "metadata/LibriTTS/1_forget_speakers/train/diffusion_data.csv"
  forget_speaker_file: "metadata/LibriTTS/1_forget_speakers/train/forget_speaker_train.csv"
  root_path:           "/path/to/LibriTTS"
  style_root_path:     "/path/to/style_vectors"
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
| `--comment` | None | Suffix appended to the W&B run name |

Checkpoints are saved to `log_dir/` as `epoch_2nd_NNNNN.pth` and `last.pth`. TensorBoard logs are written to `log_dir/<run_name>/tensorboard/`. Launch the viewer with:

```bash
tensorboard --logdir Models/Unlearning
```

Set `tensorboard.enable: false` in the config to disable logging.

Only the diffusion module is updated during training — all other components are frozen.

---

## Generating Style Vectors (optional)

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

---

## Acknowledgements

This project builds directly on [StyleTTS2](https://github.com/yl4579/StyleTTS2) by Yinghao Aaron Li et al. Please cite the original work if you use this codebase.
