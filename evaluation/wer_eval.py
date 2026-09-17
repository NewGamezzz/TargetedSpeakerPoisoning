"""Word / character error rate of the generated speech (utility metric).

Transcribes every generated wav with Whisper and compares against the manifest
transcript. Both sides pass through Whisper's ``EnglishTextNormalizer`` before
scoring, so punctuation and casing do not count against the model.

Example
-------
python evaluation/wer_eval.py \
    --inference_csv metadata/LibriTTS/15_forget_speakers/test/forget_speaker_test_test_clean.csv \
    --gen_dir       outputs/15_forget_tgu/gen_files \
    --output_dir    outputs/15_forget_tgu/eval/wer
"""

import argparse
import csv
import json
import re
import ssl
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer

from common import load_manifest, resolve_generated

MODEL_IDS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper ASR evaluation with WER/CER")
    parser.add_argument("--inference_csv", required=True, help="CSV with 'transcript' and 'speaker_files'")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated audio (gen_files)")
    parser.add_argument("--output_dir", required=True, help="Directory for the WER/CER results")
    parser.add_argument("--model_size", default="medium", choices=sorted(MODEL_IDS))
    parser.add_argument("--language", default="english")
    parser.add_argument("--max_duration_sec", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=64, help="Rows between progress saves")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def normalize_text(text, normalizer):
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(item) for item in text)
    text = unicodedata.normalize("NFKC", str(text))
    text = normalizer(text) or ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_processed(progress_file, resume):
    if not resume or not progress_file.exists():
        return set()
    with open(progress_file, "r", encoding="utf-8") as file_obj:
        return set(json.load(file_obj).get("processed_files", []))


def save_processed(progress_file, processed):
    with open(progress_file, "w", encoding="utf-8") as file_obj:
        json.dump({"processed_files": sorted(processed)}, file_obj, indent=2)


def resolve_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return "cuda:0"
    if requested == "cpu":
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_pipeline(model_size, device):
    model_id = MODEL_IDS.get(model_size, f"openai/whisper-{model_size}")
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    ssl._create_default_https_context = ssl._create_unverified_context

    print(f"Loading Whisper model: {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device.startswith("cuda") else -1,
    )
    return asr_pipe, model_id


def collect_records(df, gen_dir, resume, processed):
    """Pair every generated wav with its reference transcript."""
    records = []
    stats = {"missing_audio_rows": 0, "missing_text_rows": 0, "skipped_resume": 0}

    for _, row in df.iterrows():
        reference = str(row["transcript"]).strip()
        if not reference:
            stats["missing_text_rows"] += 1
            continue

        paths = resolve_generated(gen_dir, row)
        if not paths:
            stats["missing_audio_rows"] += 1
            continue

        for path in paths:
            key = str(path)
            if resume and key in processed:
                stats["skipped_resume"] += 1
                continue
            records.append((path, reference, key))

    return records, stats


def read_all_scores(results_csv):
    wer_scores, cer_scores = [], []
    if not results_csv.exists():
        return wer_scores, cer_scores
    with open(results_csv, "r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                wer_scores.append(float(row["wer"]))
                cer_scores.append(float(row["cer"]))
            except (TypeError, ValueError, KeyError):
                pass
    return wer_scores, cer_scores


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "wer_progress.json"
    results_csv = output_dir / "wer_cer_results.csv"
    summary_json = output_dir / "wer_cer_summary.json"

    df = load_manifest(args.inference_csv)
    processed = load_processed(progress_file, args.resume)
    records, stats = collect_records(df, args.gen_dir, args.resume, processed)

    if stats["missing_audio_rows"]:
        print(f"Warning: {stats['missing_audio_rows']} rows had no generated audio under {args.gen_dir}")
    if stats["missing_text_rows"]:
        print(f"Warning: {stats['missing_text_rows']} rows have an empty transcript")
    if stats["skipped_resume"]:
        print(f"Resume: {stats['skipped_resume']} files already processed")

    if not records:
        print("No generated audio to evaluate.")
        return

    if not args.resume or not results_csv.exists():
        with open(results_csv, "w", newline="", encoding="utf-8") as file_obj:
            csv.writer(file_obj).writerow(
                ["filename", "wer", "cer", "reference_text", "hypothesis_text"]
            )

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    asr_pipe, model_id = build_pipeline(args.model_size, device)

    normalizer = EnglishTextNormalizer()
    successful = failed = skipped_duration = 0

    with open(results_csv, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        for index, (audio_path, raw_reference, key) in enumerate(
            tqdm(records, desc="Evaluating WER"), start=1
        ):
            try:
                audio, sample_rate = sf.read(str(audio_path), dtype="float32")
                if np.ndim(audio) > 1:
                    audio = np.mean(audio, axis=1)

                if len(audio) == 0 or sample_rate <= 0:
                    failed += 1
                    continue

                if len(audio) / float(sample_rate) > args.max_duration_sec:
                    skipped_duration += 1
                    failed += 1
                    continue

                reference = normalize_text(raw_reference, normalizer)
                if not reference:
                    failed += 1
                    continue

                result = asr_pipe(
                    {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": sample_rate},
                    generate_kwargs={"language": args.language},
                )
                hypothesis = normalize_text(result.get("text", ""), normalizer)

                writer.writerow(
                    [
                        audio_path.name,
                        jiwer.wer(reference, hypothesis),
                        jiwer.cer(reference, hypothesis),
                        reference,
                        hypothesis,
                    ]
                )
                processed.add(key)
                successful += 1

            except Exception as exc:
                failed += 1
                print(f"Error processing {audio_path}: {exc}")

            if index % args.batch_size == 0:
                save_processed(progress_file, processed)

    save_processed(progress_file, processed)

    all_wer, all_cer = read_all_scores(results_csv)
    if not all_wer:
        print("No successful ASR outputs were recorded.")
        return

    summary = {
        "inference_csv": args.inference_csv,
        "gen_dir": args.gen_dir,
        "model_id": model_id,
        "language": args.language,
        "max_duration_sec": args.max_duration_sec,
        "queued_this_run": len(records),
        "successful_this_run": successful,
        "failed_this_run": failed,
        "skipped_duration_this_run": skipped_duration,
        "successful_total": len(all_wer),
        "average_wer": float(np.mean(all_wer)),
        "average_cer": float(np.mean(all_cer)),
        **stats,
    }

    with open(summary_json, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("\nWER/CER evaluation completed")
    print(f"Average WER: {summary['average_wer']:.4f}")
    print(f"Average CER: {summary['average_cer']:.4f}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
