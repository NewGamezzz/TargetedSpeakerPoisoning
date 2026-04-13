"""
Download model weights from HuggingFace.

Repository: NewGame/targeted-speaker-poisoning

Structure
---------
  pretrained/
    epochs_2nd_00020.pth       — base StyleTTS2 pretrained checkpoint
  {N}_forget_{method}/
    last.pth                   — unlearned checkpoint

Where N ∈ {1, 15, 100} and method ∈ {egu, egu_triplet, tgu, tgu_triplet}.

Usage
-----
  # Download all weights (pretrained + all unlearned models)
  python download_weights.py --output_dir Models

  # Download pretrained checkpoint only
  python download_weights.py --output_dir Models --pretrained_only

  # Download a specific setting and mode
  python download_weights.py --output_dir Models --setting 15 --mode tgu

  # Download with a HuggingFace token (for private repos)
  python download_weights.py --output_dir Models --token YOUR_HF_TOKEN
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "NewGame/targeted-speaker-poisoning"

PRETRAINED_FILE = "pretrained/epochs_2nd_00020.pth"

SETTINGS = ["1", "15", "100"]
MODES    = ["egu", "egu_triplet", "tgu", "tgu_triplet"]


def _model_folders(setting=None, mode=None):
    """Return list of (repo_folder, local_folder) tuples to download."""
    settings = [setting] if setting else SETTINGS
    modes    = [mode]    if mode    else MODES
    return [
        (f"{s}_forget_{m}", f"{s}_forget_{m}")
        for s in settings
        for m in modes
    ]


def download_weights(
    output_dir: str,
    pretrained_only: bool = False,
    setting: str = None,
    mode: str = None,
    token: str = None,
):
    """
    Download model weights from HuggingFace.

    Parameters
    ----------
    output_dir : str
        Local root directory to save weights.
    pretrained_only : bool
        Only download the pretrained base checkpoint.
    setting : str, optional
        Forget-speaker count to download: '1', '15', or '100'.
        If None, all settings are downloaded.
    mode : str, optional
        Unlearning mode: 'egu', 'egu_triplet', 'tgu', or 'tgu_triplet'.
        If None, all modes are downloaded.
    token : str, optional
        HuggingFace access token (only needed for private repositories).
    """
    def _download(repo_file, local_path, size_hint=""):
        if os.path.exists(local_path):
            print(f"  Already exists, skipping: {local_path}")
            return
        label = f"{repo_file}" + (f" ({size_hint})" if size_hint else "")
        print(f"  Downloading {label} ...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="model",
                filename=repo_file,
                local_dir=output_dir,
                token=token,
            )
            print(f"  Saved to: {local_path}")
        except Exception as e:
            print(f"  Failed: {repo_file} — {e}")

    # Pretrained checkpoint + config
    print("\n--- Pretrained ---")
    _download(
        "pretrained/epochs_2nd_00020.pth",
        os.path.join(output_dir, "pretrained", "epochs_2nd_00020.pth"),
        "~771 MB",
    )
    _download(
        "pretrained/config.yml",
        os.path.join(output_dir, "pretrained", "config.yml"),
    )

    if pretrained_only:
        return

    # Unlearned checkpoints + configs
    for repo_folder, local_folder in _model_folders(setting, mode):
        print(f"\n--- {repo_folder} ---")
        _download(
            f"{repo_folder}/last.pth",
            os.path.join(output_dir, local_folder, "last.pth"),
            "~969 MB",
        )
        _download(
            f"{repo_folder}/config.yml",
            os.path.join(output_dir, local_folder, "config.yml"),
        )

    print(f"\nDone. Weights saved to: {os.path.abspath(output_dir)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download StyleTTS2 unlearning model weights from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Models",
        help="Local directory to save model weights (default: Models/).",
    )
    parser.add_argument(
        "--pretrained_only",
        action="store_true",
        help="Download only the pretrained base checkpoint.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=SETTINGS,
        default=None,
        help="Forget-speaker count to download: 1, 15, or 100. "
             "If omitted, all settings are downloaded.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=MODES,
        default=None,
        help="Unlearning mode to download: egu, egu_triplet, tgu, tgu_triplet. "
             "If omitted, all modes are downloaded.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (only needed for private repositories).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_weights(
        output_dir=args.output_dir,
        pretrained_only=args.pretrained_only,
        setting=args.setting,
        mode=args.mode,
        token=args.token,
    )
