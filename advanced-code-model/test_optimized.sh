#!/bin/bash
# Test optimized Mamba and Hybrid architectures

cd /Users/vijaysingh/code/vijayllm/llm-from-scratch/advanced-code-model

echo "=========================================
Testing: MAMBA (OPTIMIZED)
========================================="
python3 -B scripts/train.py \
  --architecture mamba \
  --stage language \
  --model-size tiny \
  --batch-size 4 \
  --num-epochs 1 \
  --steps-per-epoch 100 \
  --state-size 16

echo ""
echo "=========================================
Testing: HYBRID (OPTIMIZED)
========================================="
python3 -B scripts/train.py \
  --architecture hybrid \
  --stage language \
  --model-size tiny \
  --batch-size 4 \
  --num-epochs 1 \
  --steps-per-epoch 100 \
  --state-size 16 \
  --hybrid-local-window 256

echo ""
echo "=========================================
RESULTS SUMMARY
========================================="
echo "Check the validation losses and training times above."
echo "Compare with previous results:"
echo "  Dense:  2:08, Val loss 5.96"
echo "  MoE:    7:36, Val loss 6.06"
echo "  Mamba (old): 8:30, Val loss 6.12"
echo "  Hybrid (old): 6.5 hours (killed)"
