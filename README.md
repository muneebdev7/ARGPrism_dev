# ARGprism

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)

**ARGprism** is a deep learning-based pipeline for predicting and annotating Antibiotic Resistance Genes (ARGs) from protein sequences.

## 🚀 Features

- **Deep Learning Classification**: Uses ProtAlbert transformer embeddings and a trained neural network to predict ARGs
- **High Accuracy**: Trained on comprehensive ARG databases
- **Fast Processing**: GPU-accelerated embedding generation
- **Comprehensive Annotation**: Maps predicted ARGs to reference databases using DIAMOND BLAST
- **Easy to Use**: Simple command-line interface
- **Flexible**: Works on CPU or GPU

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Input/Output Files](#inputoutput-files)
- [Example](#example)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Linux operating system (Ubuntu 20.04+ recommended)
- Conda or Miniconda installed
- NVIDIA GPU with CUDA 12.4+ support (optional, but recommended for speed)
- At least 8 GB RAM (16+ GB recommended)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/muneebdev7/ARGprism.git
cd ARGprism

# Create conda environment from file
conda env create -f environment_user.yml

# Activate the environment
conda activate argprism

# Verify installation
argprism --version
```

For detailed installation instructions, including troubleshooting, see [INSTALL.md](INSTALL.md).

## ⚡ Quick Start

```bash
# Activate the environment
conda activate argprism

# Run ARGprism on example data
argprism \
  -i Input_proteins/ERR589503_PROT.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  -o results/
```

## 📖 Usage

### Basic Command

```bash
argprism -i INPUT.faa -c MODEL.pth -d DATABASE.fasta -m METADATA.json -o OUTPUT_DIR
```

### Command-Line Options

| Option | Required | Description |
|--------|----------|-------------|
| `-i, --input` | Yes | Input protein sequences in FASTA format |
| `-c, --classifier` | Yes | Path to trained ARG classifier model (.pth file) |
| `-d, --database` | Yes | ARG reference database in FASTA format |
| `-m, --metadata` | Yes | ARG metadata in JSON format |
| `-o, --output` | No | Output directory (default: current directory) |
| `--cpu` | No | Force CPU usage (default: auto-detect GPU) |
| `-v, --version` | No | Show version and exit |

### Python API

You can also use ARGprism as a Python library:

```python
from argprism import ARGPrismPipeline

# Initialize pipeline
pipeline = ARGPrismPipeline(
    classifier_path="trained_model/best_model_fold4.pth",
    device="cuda"  # or "cpu"
)

# Run pipeline
pipeline.run(
    input_fasta="input.faa",
    arg_db_fasta="ARGPrismDB.fasta",
    metadata_json="metadata_arg.json",
    output_dir="results/"
)
```

## 🔬 Pipeline Overview

ARGprism follows a six-step workflow:

```
Input Sequences (.faa)
    ↓
1. Embedding Generation (ProtAlbert)
    ↓
2. ARG Classification (Neural Network)
    ↓
3. Save Predicted ARGs
    ↓
4. DIAMOND BLAST Mapping
    ↓
5. Parse Best Hits
    ↓
6. Generate Annotated Report (.csv)
```

### Step-by-Step Process

1. **Embedding Generation**: Uses ProtAlbert (a protein language model) to generate 4096-dimensional embeddings for each protein sequence

2. **Classification**: A trained neural network classifies each protein as ARG or Non-ARG

3. **ARG Extraction**: Sequences predicted as ARGs are saved to a separate FASTA file

4. **Reference Mapping**: DIAMOND BLAST maps predicted ARGs against the reference ARG database

5. **Hit Parsing**: Best hits are identified based on bit score

6. **Report Generation**: Final CSV report includes:
   - Sequence IDs
   - Predicted class (ARG/Non-ARG)
   - Reference ARG ID
   - ARG name
   - Associated drug resistance

## 📁 Input/Output Files

### Input Files

1. **Input Sequences** (`-i`): Protein sequences in FASTA format
   ```
   >seq1
   MKIVKRILLVLLSLFFTVEYSNAQTDNLTLKIENVLK...
   >seq2
   MSLYRRLVLLSCLSWPLAGFSATALTNLVAEPFAKLE...
   ```

2. **Classifier Model** (`-c`): Pre-trained PyTorch model (`.pth` file)

3. **Reference Database** (`-d`): ARG reference database in FASTA format

4. **Metadata** (`-m`): JSON file containing ARG annotations
   ```json
   {
     "ARGPrism|WP_123456|...": {
       "ARG_name": "TEM-1",
       "drug": "beta_lactam"
     }
   }
   ```

### Output Files

All outputs are saved to the specified output directory:

1. **predicted_ARGs.fasta**: Sequences predicted as ARGs
2. **predicted_ARGs_vs_ref.tsv**: DIAMOND BLAST results
3. **final_ARG_prediction_report.csv**: Annotated prediction report
4. **diamond_arg_db.dmnd**: DIAMOND database index (created automatically)

### Output Report Format

The final CSV report contains:

| Column | Description |
|--------|-------------|
| Sequence_ID | Original sequence identifier |
| Predicted_Class | ARG or Non-ARG |
| ARG_Ref_ID | Reference database match ID |
| ARG_Name | ARG gene/protein name |
| Drug | Associated antibiotic class |

## 💡 Example

### Running on Sample Data

The repository includes sample protein sequences in the `Input_proteins/` directory:

```bash
# Process a single sample
argprism \
  -i Input_proteins/ERR589503_PROT.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  -o results_ERR589503/
```

### Batch Processing Multiple Samples

```bash
# Create output directory
mkdir -p batch_results

# Process all samples
for faa in Input_proteins/*.faa; do
  sample=$(basename "$faa" _PROT.faa)
  echo "Processing $sample..."
  
  argprism \
    -i "$faa" \
    -c trained_model/best_model_fold4.pth \
    -d ARGPrismDB.fasta \
    -m metadata_arg.json \
    -o "batch_results/$sample/"
done
```

### CPU-Only Mode

If you don't have a GPU or want to use CPU:

```bash
argprism \
  -i input.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  --cpu
```

## 📊 Performance

### Speed Benchmarks

| Sequences | GPU (V100) | CPU (8 cores) |
|-----------|------------|---------------|
| 100 | ~30 sec | ~5 min |
| 1,000 | ~3 min | ~40 min |
| 10,000 | ~25 min | ~6 hours |

*Note: Times vary based on sequence length and hardware*

### Memory Requirements

- **GPU**: ~8 GB VRAM recommended
- **CPU**: ~4-8 GB RAM

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Use CPU instead
   argprism -i input.faa ... --cpu
   ```

2. **DIAMOND not found**
   ```bash
   # Reinstall DIAMOND
   conda install -c bioconda diamond=2.1.9 -y
   ```

3. **Module import errors**
   ```bash
   # Reinstall dependencies
   pip install -r argprism_requirements.txt
   ```

For more troubleshooting, see [INSTALL.md](INSTALL.md).

## 📚 Model Information

### ProtAlbert Embeddings

- **Model**: Rostlab/prot_albert
- **Architecture**: ALBERT (A Lite BERT)
- **Training**: Trained on UniRef100
- **Output**: 4096-dimensional embeddings

### ARG Classifier

- **Architecture**: Feed-forward neural network
  - Input: 4096 dimensions
  - Hidden 1: 512 neurons (ReLU)
  - Hidden 2: 128 neurons (ReLU)
  - Output: 2 classes (Softmax)
- **Training**: Cross-validated on curated ARG dataset

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact:
- **Author**: Muneeb
- **GitHub**: [@muneebdev7](https://github.com/muneebdev7)

## 🙏 Acknowledgments

- ProtAlbert model from [Rostlab](https://github.com/agemagician/ProtTrans)
- DIAMOND aligner from [Benjamin Buchfink](https://github.com/bbuchfink/diamond)
- ARG reference databases from various sources

## 📖 Citation

If you use ARGprism in your research, please cite:

```bibtex
@software{argprism2025,
  author = {Muneeb},
  title = {ARGprism: Deep Learning-based Antibiotic Resistance Gene Prediction},
  year = {2025},
  url = {https://github.com/muneebdev7/ARGprism}
}
```

---

**Version**: 1.0.0  
**Last Updated**: November 2025