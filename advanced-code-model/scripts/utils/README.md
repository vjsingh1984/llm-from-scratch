# Utility Scripts

Helper scripts and domain-specific tools.

## Domain Data Generation

```bash
# Generate domain-specific training data
python ../generate_domain_data.py

# Prepare domain data for training
python ../prepare_domain_data.py
```

## Available Domains

The `domains/` subdirectory contains generators for:

| Domain | Description |
|--------|-------------|
| `sql_domain.py` | SQL queries and database operations |
| `ml_ai_domain.py` | Machine learning code |
| `data_engineering_domain.py` | ETL and data pipelines |
| `aws_domain.py` | AWS CLI and SDK code |
| `gcp_domain.py` | Google Cloud code |
| `azure_domain.py` | Azure code |
| `devops_domain.py` | CI/CD and infrastructure |

## RAG (Retrieval-Augmented Generation)

```bash
# Build codebase index
python ../build_codebase_index.py

# Search with RAG
python ../rag_search.py --query "database connection"
```

## Multimodal

```bash
# Prepare multimodal data (code + images)
python ../prepare_multimodal_data.py
```
