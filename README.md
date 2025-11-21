# ARGPrism

[![Conda](https://img.shields.io/badge/conda-supported-brightgreen.svg?logo=anaconda&logoColor=white)](https://anaconda.org/)
[![pip](https://img.shields.io/badge/pip-installable-blue.svg?logo=pypi&logoColor=white)](https://pypi.org/)
[![GPU](https://img.shields.io/badge/GPU-Enabled-brightgreen.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/gpu-computing)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=opensource)](https://opensource.org/licenses/MIT)

**ARGPrism** is a deep learning-based pipeline for predicting and annotating Antibiotic Resistance Genes (ARGs) from protein sequences using transformer embeddings and neural networks.

## Key Features

- **Deep Learning Classification**: ProtAlbert transformer embeddings + neural network classifier
- **GPU Accelerated**: Fast processing with CUDA support  
- **Reference Mapping**: DIAMOND BLAST alignment to ARG databases
- **Simple Interface**: Easy-to-use command line tool
- **Flexible Deployment**: CPU or GPU execution

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Input/Output](#inputoutput)
- [License](#license)

## Installation

### Prerequisites

- Linux operating system (Ubuntu 20.04+)
- Conda/Miniconda/**Mamba (Recommended)** must be installed
- 8+ GB RAM (16 GB recommended)
- NVIDIA GPU with CUDA 11.8+ or 12.x (optional, for acceleration)

### Option 1: Install from Conda (Recommended)

```bash
# Install from conda-forge
conda install -c conda-forge argprism

# Verify installation
python -m argprism --version
```

### Option 2: Install from Source

```bash
# Clone repository
git clone https://github.com/haseebmanzur/ARGPrism.git
cd ARGprism

# Create environment
mamba env create -f environment.yml

# Activate environment  
mamba activate argprism

# Verify installation
argprism --version
```

## Quick Start

```bash
# Activate environment
conda activate argprism

# Run on test data
python -m argprism Test_dataset/ERR589441_PROT_sampled.faa --output-dir results/
```

## Usage

### Command Line

```bash
python -m argprism INPUT_FILE.faa [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory | `argprism_output` |
| `--device` | Force CPU/CUDA usage | Auto-detect |
| `--quiet` | Reduce output verbosity | False |

### Python API

```python
from argprism import run_pipeline

# Run pipeline
result = run_pipeline(
    input_fasta="input.faa",
    output_dir="results/",
    verbose=True
)

print(f"Predictions: {len(result.predictions)}")
print(f"ARGs found: {result.predicted_fasta}")
```

## Pipeline Overview

ARGPrism processes protein sequences through the following steps:

```text
Input FASTA → ProtAlbert Embeddings → Neural Classifier → ARG Prediction → DIAMOND Mapping → Report
```

### Process Details

1. **Embedding Generation**: ProtAlbert generates 4096-dimensional embeddings
2. **Classification**: Neural network predicts ARG/Non-ARG for each sequence  
3. **Reference Mapping**: DIAMOND aligns predicted ARGs to reference database
4. **Report Generation**: Creates annotated CSV with ARG names and drug classes

## Input/Output

### Input

- **FASTA file**: Protein sequences to analyze
- Built-in models and databases are included

### Output Files

All results saved to output directory:

- `predicted_ARGs.fasta` - Sequences classified as ARGs
- `predicted_ARGs_vs_ref.tsv` - DIAMOND alignment results  
- `final_ARG_prediction_report.csv` - Annotated predictions with ARG names/drugs
- `diamond_arg_db.dmnd` - DIAMOND database index

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on GitHub.

**Author**: Haseeb Manzoor  
**GitHub**: [@haseebmanzur](https://github.com/haseebmanzur)

**Package Maintainer**: Muhammad Muneeb Nasir  
**GitHub**: [@muneebdev7](https://github.com/muneebdev7)

## Acknowledgments

- [ProtAlbert](https://github.com/agemagician/ProtTrans) - Protein language model
- [DIAMOND](https://github.com/bbuchfink/diamond) - Sequence alignment tool

## Citation

If you use ARGPrism in your research, please cite:
