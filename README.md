# StyleTTS2 Speaker Unlearning

Fine-tuning framework for **machine unlearning** applied to [StyleTTS2](https://github.com/yl4579/StyleTTS2). Given a pretrained StyleTTS2 checkpoint, this codebase fine-tunes the diffusion module so the model forgets a specific speaker's voice while retaining synthesis quality for all other speakers.

## Requirements

```bash
pip install -r requirements.txt
```

You also need:
- A pretrained **StyleTTS2 second-stage checkpoint**. Download the LibriTTS model from [https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main). Place the files according to the paths in the config:
  - Epoch checkpoint → `Models/LibriTTS/epochs_2nd_00020.pth`

## Data Preparation

All metadata files and pre-computed style vectors are available on HuggingFace:
**[Dataset: YOUR_HUGGINGFACE_DATASET_LINK]**

The raw audio must be downloaded separately from [LibriTTS](https://www.openslr.org/60/).

### Dataset settings

Three experimental settings are provided, varying the number of forget speakers:

| Setting | Forget speakers |
|---------|----------------|
| `1_speaker` | 1 |
| `15_speakers` | 15 |
| `100_speakers` | 100 |

Select a setting by pointing `forget_speaker_file` in the config to the corresponding CSV (e.g. `metadata/settings/1_speaker/forget_train.csv`). The training data (`diffusion_data.csv`) and style vectors are shared across all settings.

### Training data CSV (`dataset_path`)

| Column | Description |
|--------|-------------|
| `filepath` | Relative path to a pre-computed style vector (`.pt`), resolved against `style_root_path` |
| `text` | Phonemised transcript |
| `speaker_id` | Speaker identifier |
| `ref_filepath` | Relative path to the reference `.wav` for this utterance, resolved against `root_path` |

### Forget-speaker CSV (`forget_speaker_file`)

| Column | Description |
|--------|-------------|
| `speaker_files` | Relative path to a `.wav` file for the speaker to be forgotten, resolved against `root_path` |
| `speaker_ids` | Speaker identifier |

### Inference / evaluation CSV (`--inference_csv`)

| Column | Description |
|--------|-------------|
| `transcript` | Text to synthesize |
| `speaker_files` | Relative path to the reference audio (`.wav`), resolved against `root_path` |


## Generating Style Vectors (optional)

Pre-computed style vectors are provided on HuggingFace. If you prefer to generate them yourself from a pretrained checkpoint, use `gen_diffusion_ground_truth.py`:

```bash
python gen_diffusion_ground_truth.py \
    --input_csv      metadata/retain_speaker_train.csv \
    --config_path    Configs/config_unlearning.yml \
    --checkpoint_path Models/LibriTTS/epochs_2nd_00020.pth \
    --root_path      /path/to/LibriTTS \
    --output_dir     style_vectors \
    --diffusion_samples 2 \
    --resume
```

This reads each row from the input CSV, runs the diffusion sampler to produce style tensors, and writes them as `.pt` files under `style_vectors/diffusion/` and `style_vectors/ref/`. It also generates `diffusion_data.csv` and `ref_data.csv` with relative paths ready to use as `dataset_path` in the config.

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_csv` | required | CSV with `speaker_files`, `transcript`, `speaker_ids`, `output_files` columns |
| `--config_path` | `Configs/config_unlearning.yml` | Path to the StyleTTS2 config YAML |
| `--checkpoint_path` | `Models/LibriTTS/epochs_2nd_00020.pth` | Path to the pretrained checkpoint |
| `--root_path` | required | Root directory for LibriTTS `.wav` files |
| `--output_dir` | `style_vectors` | Directory to write `.pt` files and metadata CSVs |
| `--diffusion_samples` | `2` | Number of independently sampled diffusion vectors per utterance |
| `--diffusion_steps` | `5` | Number of diffusion sampling steps |
| `--resume` | off | Skip utterances whose output `.pt` files already exist |

---

## Configuration

Edit [Configs/config_unlearning.yml](Configs/config_unlearning.yml) before training. Key fields:

```yaml
pretrained_model: "Models/LibriTTS/epochs_2nd_00020.pth"  # Path to 2nd-stage checkpoint
log_dir: "Models/Unlearning"          # Output directory for checkpoints and logs

data_params:
  dataset_path: "/path/to/diffusion_data.csv"
  forget_speaker_file: "/path/to/forget_speaker_train.csv"
  forget_ratio: 0.6          # Probability of replacing reference with a forget-speaker sample
  root_path: "/path/to/LibriTTS/wavs"      # root for LibriTTS audio files
  style_root_path: "/path/to/style_vectors" # root for pre-computed style vectors

epochs: 10
batch_size: 32

loss_params:
  lambda_triplet: 1.0        # Triplet loss weight (triplet mode only)
  margin: 0.3                # Triplet margin (Euclidean distance)
```

All model architecture fields under `model_params` must match the pretrained checkpoint.


## Training

TGU / EGU

```bash
python train.py --config Configs/config_unlearning.yml --mode standard
```

Triplet variants

```bash
python train.py --config Configs/config_unlearning.yml --mode triplet \
                --lambda_triplet 1.0
```

**All CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `Configs/config_unlearning.yml` | Path to YAML config |
| `--mode` | `standard` | `standard` or `triplet` |
| `--forget_ratio` | from config | Probability [0, 1] of substituting a forget-speaker reference during training |
| `--lambda_triplet` | from config | Weight for the triplet loss (triplet mode only) |
| `--max_iter` | 60,000 | Train for exactly this many gradient steps, cycling the dataloader as needed. When set, epoch count from config is ignored. |
| `--comment` | None | Optional suffix appended to the W&B run name |

Checkpoints are saved to `log_dir/` every `save_freq` epochs as `epoch_2nd_NNNNN.pth`, plus a rolling `last.pth`. Training metrics are logged to [Weights & Biases](https://wandb.ai); set `wandb.enable: false` in the config to disable.

## Inference

```bash
python infer.py \
    --config   path/to/run_dir/config_unlearning.yml \
    --checkpoint path/to/run_dir/last.pth \
    --inference_csv path/to/test.csv \
    --output_dir ./outputs \
    --utterance_samples 5 \
    --diffusion_samples 3
```

Outputs are written to:
```
outputs/
  ref_files/   — copies of the reference audio files
  gen_files/
    <speaker_stem>/
      sample_0.wav
      sample_1.wav
      ...
```

**All CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Path to the config YAML |
| `--checkpoint` | required | Path to checkpoint (`.pth`) |
| `--inference_csv` | required | CSV with `transcript` and `speaker_files` columns |
| `--output_dir` | `./outputs` | Directory for generated audio |
| `--utterance_samples` | `0` (all) | Number of rows to randomly sample from the CSV |
| `--diffusion_samples` | `1` | Independent waveforms to generate per utterance |
| `--alpha` | `1.0` | Style-encoder interpolation weight (0 = full reference, 1 = full predicted) |
| `--beta` | `1.0` | Predictor-encoder interpolation weight (0 = full reference, 1 = full predicted) |
| `--diffusion_steps` | `5` | Number of diffusion sampling steps |
| `--seed` | `0` | Random seed for utterance sampling |

**Alpha / Beta:** Setting both to `1.0` means the model relies entirely on its diffusion-predicted style and is the standard evaluation setup for unlearning assessment. To evaluate how well the forget-speaker's voice has been erased, point `--inference_csv` at the forget-speaker's audio and compare the generated voice against the original.


## Acknowledgements

This project builds directly on [StyleTTS2](https://github.com/yl4579/StyleTTS2) by Yinghao Aaron Li et al. Please cite the original work if you use this codebase.
