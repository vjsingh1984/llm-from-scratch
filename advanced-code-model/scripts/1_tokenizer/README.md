# Stage 1: Tokenizer Training

Train the BPE tokenizer on your code corpus.

## Quick Start

```bash
# Train tokenizer on code data
python ../train_code_tokenizer.py

# Or train basic tokenizer
python ../train_tokenizer.py
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `train_tokenizer.py` | Basic BPE tokenizer training |
| `train_code_tokenizer.py` | Code-optimized tokenizer |

## Output

Tokenizer files are saved to `data/tokenizer/`:
- `tokenizer.json` - Main tokenizer file
- `vocab.json` - Vocabulary mapping

## Next Step

After training tokenizer, proceed to [Stage 2: Language Pretraining](../2_pretrain/).
