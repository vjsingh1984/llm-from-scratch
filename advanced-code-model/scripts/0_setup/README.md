# Stage 0: Environment Setup

Scripts for downloading data and setting up the environment.

## Quick Start

```bash
# Download sample data (small, for testing)
python ../download_sample_data.py

# Download full datasets
python ../download_code_corpus.py
python ../download_bash_corpus.py

# Or use the setup script
bash ../setup_data.sh
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `download_sample_data.py` | Download small sample for testing |
| `download_code_corpus.py` | Download Python/code corpus |
| `download_bash_corpus.py` | Download bash script corpus |
| `download_more_bash.py` | Additional bash scripts |
| `download_bookcorpus.py` | Download BookCorpus for language |
| `setup_data.sh` | All-in-one setup script |

## Next Step

After downloading data, proceed to [Stage 1: Tokenizer](../1_tokenizer/).
