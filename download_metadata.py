"""
Download metadata and style vectors from HuggingFace.

Each setting lives in its own HuggingFace repository with the structure:
  train/
    diffusion_data.csv
    forget_speaker_train.csv
    ref_data.csv
    retain_speaker_train.csv
    style_vectors.zip
  test/
    forget_speaker_test.csv
    forget_speaker_test_test_clean.csv
    retain_speaker_test.csv
    retain_speaker_test_test_clean.csv

Usage
-----
  # Download all settings (metadata + style vectors)
  python download_metadata.py --output_dir metadata

  # Download a specific setting only
  python download_metadata.py --output_dir metadata --setting 15_forget_speakers

  # Download metadata CSVs only (skip style vectors)
  python download_metadata.py --output_dir metadata --no_style_vectors

  # Download with a HuggingFace token (for private repos)
  python download_metadata.py --output_dir metadata --token YOUR_HF_TOKEN
"""

import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download

# HuggingFace repo ID for each setting
REPOS = {
    "1_forget_speakers":   "NewGame/libritts-1-forget-speaker",
    "15_forget_speakers":  "NewGame/libritts-15-forget-speaker",
    "100_forget_speakers": "NewGame/libritts-100-forget-speaker",
}

SETTINGS = list(REPOS.keys())

TRAIN_FILES = [
    "train/diffusion_data.csv",
    "train/forget_speaker_train.csv",
    "train/ref_data.csv",
    "train/retain_speaker_train.csv",
]

TEST_FILES = [
    "test/forget_speaker_test.csv",
    "test/forget_speaker_test_test_clean.csv",
    "test/retain_speaker_test.csv",
    "test/retain_speaker_test_test_clean.csv",
]

STYLE_VECTORS_ZIP = "train/style_vectors.zip"


def download_metadata(
    output_dir: str,
    setting: str = None,
    download_style_vectors: bool = True,
    token: str = None,
):
    """
    Download metadata CSVs (and optionally style vectors) from HuggingFace.

    Files are placed under:
      <output_dir>/LibriTTS/<setting>/train/
      <output_dir>/LibriTTS/<setting>/test/

    Style vectors are extracted to:
      <output_dir>/LibriTTS/<setting>/train/style_vectors/

    Parameters
    ----------
    output_dir : str
        Local root directory for all downloaded files.
    setting : str, optional
        One specific setting to download. If None, all settings are downloaded.
    download_style_vectors : bool
        Whether to download and extract style_vectors.zip (default: True).
    token : str, optional
        HuggingFace access token (only needed for private repositories).
    """
    settings_to_download = [setting] if setting else SETTINGS

    for s in settings_to_download:
        repo_id = REPOS[s]
        local_base = os.path.join(output_dir, "LibriTTS", s)
        print(f"\n--- Setting: {s}  (repo: {repo_id}) ---")

        # Download CSV files
        for repo_file in TRAIN_FILES + TEST_FILES:
            local_path = os.path.join(local_base, repo_file)

            if os.path.exists(local_path):
                print(f"  Already exists, skipping: {local_path}")
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=repo_file,
                    local_dir=local_base,
                    token=token,
                )
                print(f"  Downloaded: {repo_file}")
            except Exception as e:
                print(f"  Failed: {repo_file} — {e}")

        # Download and extract style vectors
        if download_style_vectors:
            zip_local = os.path.join(local_base, STYLE_VECTORS_ZIP)
            extract_dir = os.path.join(local_base, "train")

            if os.path.exists(os.path.join(extract_dir, "style_vectors")):
                print(f"  Style vectors already extracted, skipping.")
            else:
                print(f"  Downloading style_vectors.zip (~2.3 GB) ...")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=STYLE_VECTORS_ZIP,
                        local_dir=local_base,
                        token=token,
                    )
                    print(f"  Extracting style_vectors.zip ...")
                    with zipfile.ZipFile(zip_local, "r") as zf:
                        zf.extractall(extract_dir)
                    os.remove(zip_local)
                    print(f"  Extracted to: {extract_dir}")
                except Exception as e:
                    print(f"  Failed to download/extract style_vectors.zip — {e}")

    print(f"\nDone. Files saved to: {os.path.abspath(output_dir)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download StyleTTS2 unlearning metadata from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metadata",
        help="Local directory to save files (default: metadata/).",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=SETTINGS,
        default=None,
        help="Download only this setting. If omitted, all three settings are downloaded.",
    )
    parser.add_argument(
        "--no_style_vectors",
        action="store_true",
        help="Skip downloading style_vectors.zip.",
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
    download_metadata(
        output_dir=args.output_dir,
        setting=args.setting,
        download_style_vectors=not args.no_style_vectors,
        token=args.token,
    )
