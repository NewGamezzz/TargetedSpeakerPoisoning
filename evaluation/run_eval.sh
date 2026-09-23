#!/usr/bin/env bash
# End-to-end evaluation for one unlearned model.
#
# Runs inference on the retain and forget test sets, then computes every metric
# in the paper: WER, UTMOS, SSIM (both sets), AUC, Avg-FSSIM and Max-FSSIM.
#
# Usage:
#   bash evaluation/run_eval.sh <model_dir> <metadata_setting_dir> <libritts_root> <output_dir>
#
# Example:
#   bash evaluation/run_eval.sh \
#       Models/15_forget_tgu \
#       metadata/LibriTTS/15_forget_speakers \
#       /path/to/LibriTTS \
#       outputs/15_forget_tgu

set -euo pipefail

MODEL_DIR=${1:?model_dir is required}
META_DIR=${2:?metadata setting directory is required}
ROOT_PATH=${3:?LibriTTS root is required}
OUT_DIR=${4:?output_dir is required}

RETAIN_CSV="${META_DIR}/test/retain_speaker_test_test_clean.csv"
FORGET_CSV="${META_DIR}/test/forget_speaker_test_test_clean.csv"
EVAL_DIR="${OUT_DIR}/eval"

echo "=== 1/6  Inference ==============================================="
for subset in retain forget; do
    csv_var="$(echo "$subset" | tr '[:lower:]' '[:upper:]')_CSV"
    python infer.py \
        --model_dir     "$MODEL_DIR" \
        --inference_csv "${!csv_var}" \
        --root_path     "$ROOT_PATH" \
        --output_dir    "${OUT_DIR}/${subset}"
done

echo "=== 2/6  WER (Whisper-medium) ===================================="
# The paper reports WER for both the retain and the forget set.
for subset in retain forget; do
    csv_var="$(echo "$subset" | tr '[:lower:]' '[:upper:]')_CSV"
    python evaluation/wer_eval.py \
        --inference_csv "${!csv_var}" \
        --gen_dir       "${OUT_DIR}/${subset}/gen_files" \
        --output_dir    "${EVAL_DIR}/wer_${subset}"
done

echo "=== 3/6  UTMOS =================================================="
# Likewise MOS, which the paper reports for both sets.
for subset in retain forget; do
    python evaluation/mos_eval.py \
        --gen_dir     "${OUT_DIR}/${subset}/gen_files" \
        --output_file "${EVAL_DIR}/utmos_${subset}.txt"
done

echo "=== 4/6  SSIM (retain and forget) ==============================="
for subset in retain forget; do
    csv_var="$(echo "$subset" | tr '[:lower:]' '[:upper:]')_CSV"
    python evaluation/ssim_eval.py \
        --inference_csv "${!csv_var}" \
        --gen_dir       "${OUT_DIR}/${subset}/gen_files" \
        --root_path     "$ROOT_PATH" \
        --output_dir    "${EVAL_DIR}/ssim_${subset}"
done

echo "=== 5/6  AUC ===================================================="
python evaluation/auc_eval.py \
    --retain_csv  "${EVAL_DIR}/ssim_retain/speaker_similarity_results.csv" \
    --forget_csv  "${EVAL_DIR}/ssim_forget/speaker_similarity_results.csv" \
    --output_file "${EVAL_DIR}/auc.json"

echo "=== 6/6  FSSIM =================================================="
python evaluation/compute_embedding.py \
    --inference_csv "$FORGET_CSV" \
    --root_path     "$ROOT_PATH" \
    --output_path   "${EVAL_DIR}/forget_speaker_embeddings.npy"

python evaluation/fssim_eval.py \
    --inference_csv   "$FORGET_CSV" \
    --gen_dir         "${OUT_DIR}/forget/gen_files" \
    --embeddings_file "${EVAL_DIR}/forget_speaker_embeddings.npy" \
    --output_file     "${EVAL_DIR}/fssim.csv"

echo
echo "Done. All results are under ${EVAL_DIR}/"
