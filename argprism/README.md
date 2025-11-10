# ARGprism Python Package

This directory contains the main ARGprism Python package.

## Structure

```
argprism/
├── __init__.py          # Package initialization
├── classifier.py        # Neural network classifier
├── embeddings.py        # ProtAlbert embedding generation
├── pipeline.py          # Main pipeline logic
├── cli.py              # Command-line interface
├── data/               # Data files (to be populated)
│   ├── ARGPrismDB.fasta
│   └── metadata_arg.json
└── models/             # Model files (to be populated)
    └── best_model_fold4.pth
```

## Before Building Package

**IMPORTANT**: Copy data files to this package structure:

```bash
# From the root directory, run:
cp ARGPrismDB.fasta argprism/data/
cp metadata_arg.json argprism/data/
cp trained_model/best_model_fold4.pth argprism/models/
```

These files will be included in the package distribution via MANIFEST.in

## Modules

### `__init__.py`
- Package version and exports
- Main entry points for Python API

### `classifier.py`
- `ARGClassifier`: Neural network model
- Architecture: 4096 → 512 → 128 → 2
- Trained on ARG dataset

### `embeddings.py`
- `load_protalbert_model()`: Load ProtAlbert from HuggingFace
- `generate_embedding()`: Generate embedding for single sequence
- `generate_embeddings()`: Process entire FASTA file

### `pipeline.py`
- `ARGPrismPipeline`: Main pipeline class
- Orchestrates: embedding → classification → mapping → reporting
- Handles DIAMOND BLAST integration

### `cli.py`
- Command-line interface
- Entry point: `argprism` command
- Argument parsing and validation

## Usage

### As Python API

```python
from argprism import ARGPrismPipeline

pipeline = ARGPrismPipeline(
    classifier_path="models/best_model_fold4.pth",
    device="cuda"
)

pipeline.run(
    input_fasta="input.faa",
    arg_db_fasta="data/ARGPrismDB.fasta",
    metadata_json="data/metadata_arg.json",
    output_dir="results/"
)
```

### As CLI

```bash
argprism -i input.faa -c model.pth -d db.fasta -m meta.json -o results/
```

## Development

To test during development:

```bash
# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Version

Current version: 1.0.0

See `__init__.py` for version information.
