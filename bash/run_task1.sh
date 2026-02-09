#!/bin/bash
# Task1: Gene Regulation Direction Prediction with GAT
# This script runs the unified pipeline for task1 (upregulated/downregulated prediction)

# Configuration
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
VLLM_PORT=${VLLM_PORT:-8000}
SEED=${SEED:-0}
ORGANS=${ORGANS:-"peripheral_blood"}  # Available: bone_marrow, breast, cervix, colon, lung, peripheral_blood, prostate, skin
USE_HISTORY=${USE_HISTORY:-"true"}  # Set to "true" to enable STRING DB similarity-based history context
HISTORY_TOPK=${HISTORY_TOPK:-3}  # Number of top STRING-similar history genes to include

# Parse organs string into array
IFS=' ' read -ra ORGAN_ARRAY <<< "$ORGANS"

echo "=============================================="
echo "Task1: Gene Regulation Direction Prediction"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "VLLM Port: $VLLM_PORT"
echo "Seed: $SEED"
echo "Organs: ${ORGAN_ARRAY[*]}"
echo "Use History: $USE_HISTORY"
echo "History TopK: $HISTORY_TOPK"
echo "=============================================="

# Build command
CMD="python run.py \
    --task_type task1 \
    --use_gat \
    --organs ${ORGAN_ARRAY[*]} \
    --model_name $MODEL_NAME \
    --vllm_port $VLLM_PORT \
    --seed $SEED \
    --use_sorted_pathway \
    --batch_size 2 \
    --max_retry 2 \
    --temperature 0.6 \
    --max_tokens 4096 \
    --max_workers 16 \
    --max_pathways 10 \
    --gat_checkpoint_dir checkpoints/260119_gat_all_organs \
    --gat_device cpu \
    --sorted_pathway_dir data/lincsqa_small/gene_regulation_dir_pred/combined_score \
    --use_depmap \
    --history_topk $HISTORY_TOPK"

# Add --use_history flag if enabled
if [ "$USE_HISTORY" = "true" ]; then
    CMD="$CMD --use_history"
fi

eval $CMD

echo "=============================================="
echo "Task1 completed. Results saved to results/"
echo "=============================================="
