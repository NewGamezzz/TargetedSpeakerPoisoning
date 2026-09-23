"""Word / character error rate of the generated speech (utility metric).

Transcribes every generated wav with Whisper and compares against the manifest
transcript. Both sides pass through Whisper's ``EnglishTextNormalizer`` before
scoring, so punctuation and casing do not count against the model.

Three exclusions are applied, matching the paper:

* utterances longer than ``--max_duration_sec`` (30s) are not transcribed
* references shorter than ``--min_ref_words`` (3 words) are dropped, because a
  single substitution against a two-word reference is already a 100% WER
* hypotheses implying more than ``--max_words_per_sec`` (6) words per second of
  generated audio are dropped as degenerate synthesis

Both matter. On the 15-speaker seen setting the short-reference rule moves the
reported WER by about 1.6 points; on a model whose forget-set output has
collapsed it is the speaking-rate rule that does the work. The summary reports
the unfiltered average too, so the effect of both is always visible.

The reported ``average_wer`` is the mean of the per-utterance rates, which is
what the paper reports. ``corpus_wer`` pools every edit over every reference
word instead -- the usual ASR convention -- and is written alongside it.

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
    parser.add_argument(
        "--min_ref_words",
        type=int,
        default=3,
        help="Drop utterances whose reference is shorter than this (default 3, "
             "i.e. discard 1-2 word references). Pass 0 to keep everything.",
    )
    parser.add_argument(
        "--max_words_per_sec",
        type=float,
        default=6.0,
        help="Drop utterances whose hypothesis exceeds this speaking rate, which "
             "marks degenerate synthesis (default 6). Pass 0 to disable.",
    )
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
    """Return every scored row, with ``wer``/``cer`` parsed to float.

    Whole rows are returned rather than bare scores because the summary needs
    the text for the corpus-level rates and the word counts for the filters.
    """
    rows = []
    if not results_csv.exists():
        return rows
    with open(results_csv, "r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                row["wer"] = float(row["wer"])
                row["cer"] = float(row["cer"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def keep_row(row, min_ref_words, max_words_per_sec):
    """Apply the paper's two post-hoc exclusions to one scored utterance.

    A one- or two-word reference turns a single substitution into a 100% WER, so
    those rows dominate an unweighted mean without saying much about the model.
    A hypothesis that implies an implausible speaking rate marks synthesis that
    has collapsed into babble; the rate is measured against the generated audio,
    so it only fires on genuinely degenerate output.
    """
    reference = str(row.get("reference_text") or "")
    hypothesis = str(row.get("hypothesis_text") or "")

    ref_words = int(row["ref_words"]) if row.get("ref_words") else len(reference.split())
    if min_ref_words and ref_words < min_ref_words:
        return False

    if max_words_per_sec:
        hyp_words = int(row["hyp_words"]) if row.get("hyp_words") else len(hypothesis.split())
        try:
            duration = float(row.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0 and hyp_words > duration * max_words_per_sec:
            return False

    return True


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
                [
                    "filename", "wer", "cer", "reference_text", "hypothesis_text",
                    "duration_sec", "ref_words", "hyp_words",
                ]
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
                        len(audio) / float(sample_rate),
                        len(reference.split()),
                        len(hypothesis.split()),
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

    all_rows = read_all_scores(results_csv)
    if not all_rows:
        print("No successful ASR outputs were recorded.")
        return

    # The headline figures exclude the two categories the paper excludes; the
    # unfiltered ones are kept alongside so the effect of that choice is visible.
    kept = [r for r in all_rows if keep_row(r, args.min_ref_words, args.max_words_per_sec)]
    dropped_short = sum(
        1 for r in all_rows if not keep_row(r, args.min_ref_words, 0)
    )
    dropped_fast = sum(
        1 for r in all_rows if not keep_row(r, 0, args.max_words_per_sec)
    )
    if not kept:
        print(
            "Every utterance was excluded by the filters; reporting unfiltered "
            "figures instead. Lower --min_ref_words if the transcripts have no "
            "whitespace (e.g. Mandarin)."
        )
        kept = all_rows

    all_wer = [r["wer"] for r in kept]
    all_cer = [r["cer"] for r in kept]
    unfiltered_wer = [r["wer"] for r in all_rows]
    unfiltered_cer = [r["cer"] for r in all_rows]

    # Two aggregations are in circulation and they differ a lot on this data:
    #   average_*  mean of the per-utterance rates, so a one-word clip weighs
    #              as much as a fifty-word one
    #   corpus_*   total edits over total reference length, the usual ASR
    #              convention, which is dominated by the longer utterances
    # Both are reported so results stay comparable either way.
    pooled = [
        (str(r.get("reference_text") or ""), str(r.get("hypothesis_text") or ""))
        for r in kept
    ]
    pooled = [(r, h) for r, h in pooled if r.strip()]
    corpus_wer = corpus_cer = None
    if pooled:
        refs = [r for r, _ in pooled]
        hyps = [h for _, h in pooled]
        corpus_wer = float(jiwer.wer(refs, hyps))
        corpus_cer = float(jiwer.cer(refs, hyps))

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
        "successful_total": len(all_rows),
        "scored_after_filtering": len(all_wer),
        "min_ref_words": args.min_ref_words,
        "max_words_per_sec": args.max_words_per_sec,
        "dropped_short_reference": dropped_short,
        "dropped_fast_hypothesis": dropped_fast,
        "average_wer": float(np.mean(all_wer)),
        "average_cer": float(np.mean(all_cer)),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "average_wer_unfiltered": float(np.mean(unfiltered_wer)),
        "average_cer_unfiltered": float(np.mean(unfiltered_cer)),
        **stats,
    }

    with open(summary_json, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("\nWER/CER evaluation completed")
    print(
        f"Scored {len(all_wer)} of {len(all_rows)} utterances "
        f"({dropped_short} dropped: reference under {args.min_ref_words} words, "
        f"{dropped_fast} dropped: over {args.max_words_per_sec} words/sec)"
    )
    print(f"Average WER: {summary['average_wer']:.4f}  (mean of per-utterance rates)")
    print(f"Average CER: {summary['average_cer']:.4f}")
    if corpus_wer is not None:
        print(f"Corpus  WER: {corpus_wer:.4f}  (total edits / total reference words)")
        print(f"Corpus  CER: {corpus_cer:.4f}")
    print(f"Unfiltered average WER: {summary['average_wer_unfiltered']:.4f}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
