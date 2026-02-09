#!/bin/bash
# Task2: MoA Prediction Case Study without GAT
# This script runs the unified pipeline for task2 (yes/no or yes/no/uncertain prediction)
# Note: Task2 does not use organ-level data, only case_study

# Configuration
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
VLLM_PORT=${VLLM_PORT:-8000}
SEED=${SEED:-0}
CASE_STUDY=${CASE_STUDY:-"case1_braf"}  # Options: case1_braf, case2_kras
ALLOW_UNCERTAIN=${ALLOW_UNCERTAIN:-"false"}  # Set to "true" to allow uncertain answers
USE_HISTORY=${USE_HISTORY:-"true"}  # Set to "true" to enable STRING DB similarity-based history context
HISTORY_TOPK=${HISTORY_TOPK:-3}  # Number of top STRING-similar history genes to include

echo "=============================================="
echo "Task2: MoA Prediction Case Study"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "VLLM Port: $VLLM_PORT"
echo "Seed: $SEED"
echo "Case Study: $CASE_STUDY"
echo "Allow Uncertain: $ALLOW_UNCERTAIN"
echo "Use History: $USE_HISTORY"
echo "History TopK: $HISTORY_TOPK"
echo "=============================================="

# Build command with optional flags
# Note: --organs is required but ignored for task2, using placeholder "case_study"
CMD="python run.py \
    --task_type task2 \
    --no_gat \
    --case_study $CASE_STUDY \
    --organs case_study \
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
    --sorted_pathway_dir data/lincsqa_small/case_study/sorted/combined_score \
    --use_depmap \
    --history_topk $HISTORY_TOPK"

if [ "$ALLOW_UNCERTAIN" = "true" ]; then
    CMD="$CMD --allow_uncertain"
fi

if [ "$USE_HISTORY" = "true" ]; then
    CMD="$CMD --use_history"
fi

eval $CMD

echo "=============================================="
echo "Task2 completed. Results saved to results/"
echo "=============================================="
