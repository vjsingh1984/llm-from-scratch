#!/bin/bash
# Data Setup Script
# Downloads and prepares all datasets for training

set -e  # Exit on error

echo "============================================================"
echo "LLM Training Data Setup"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Download TinyStories dataset (~100MB)"
echo "  2. Download bash scripts for code training"
echo "  3. Train tokenizer"
echo "  4. Prepare all training data"
echo ""
echo "Total download size: ~150-200MB"
echo "Total disk space needed: ~500MB after processing"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/tokenizer

# Step 1: Download TinyStories
echo ""
echo "============================================================"
echo "Step 1: Downloading TinyStories dataset"
echo "============================================================"

if [ ! -f "data/raw/TinyStoriesV2-GPT4-train.txt" ]; then
    echo "Downloading TinyStories training data..."
    curl -L "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
        -o "data/raw/TinyStoriesV2-GPT4-train.txt"
    echo "✓ Downloaded training data"
else
    echo "✓ TinyStories training data already exists"
fi

if [ ! -f "data/raw/TinyStoriesV2-GPT4-valid.txt" ]; then
    echo "Downloading TinyStories validation data..."
    curl -L "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" \
        -o "data/raw/TinyStoriesV2-GPT4-valid.txt"
    echo "✓ Downloaded validation data"
else
    echo "✓ TinyStories validation data already exists"
fi

# Step 2: Download bash scripts
echo ""
echo "============================================================"
echo "Step 2: Downloading bash scripts for code training"
echo "============================================================"

echo "NOTE: The bash scripts dataset requires cloning multiple repos."
echo "This is handled by the data preparation script."
echo "You can skip this if you want to use the provided sample dataset."
echo ""

# Step 3: Train tokenizer
echo ""
echo "============================================================"
echo "Step 3: Training BPE tokenizer"
echo "============================================================"

if [ ! -f "data/tokenizer/tokenizer.json" ]; then
    echo "Training tokenizer on TinyStories..."
    python3 scripts/train_tokenizer.py
    echo "✓ Tokenizer trained"
else
    echo "✓ Tokenizer already exists"
fi

# Step 4: Prepare language data
echo ""
echo "============================================================"
echo "Step 4: Preparing language dataset"
echo "============================================================"

if [ ! -f "data/processed/language_train.npy" ]; then
    echo "Processing TinyStories data..."
    python3 scripts/download_data.py
    echo "✓ Language data prepared"
else
    echo "✓ Language data already prepared"
fi

# Step 5: Prepare code data
echo ""
echo "============================================================"
echo "Step 5: Preparing code dataset (bash scripts)"
echo "============================================================"

echo "This step requires cloning bash script repositories."
echo "It may take 10-15 minutes and requires ~500MB disk space."
echo ""
read -p "Download and prepare bash scripts? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "data/processed/code_train.npy" ]; then
        echo "Downloading bash scripts..."
        # The download_data.py script handles both language and code
        # If it's already run for language, it should skip that part
        python3 scripts/download_data.py
        echo "✓ Code data prepared"
    else
        echo "✓ Code data already prepared"
    fi
else
    echo "Skipped code data preparation."
    echo "You can run it later with: python3 scripts/download_data.py"
fi

# Summary
echo ""
echo "============================================================"
echo "✓ Data Setup Complete!"
echo "============================================================"
echo ""
echo "Prepared datasets:"
ls -lh data/processed/*.npy 2>/dev/null || echo "  (Run data preparation scripts to generate)"
echo ""
echo "Next steps:"
echo "  1. Verify data: ls -lh data/processed/"
echo "  2. Start Stage 1 training: python3 scripts/train.py --stage language"
echo ""
echo "See docs/STEP_BY_STEP_GUIDE.md for detailed instructions."
echo ""
