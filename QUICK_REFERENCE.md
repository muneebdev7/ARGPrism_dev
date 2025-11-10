# ARGprism Conda Package - Quick Reference

## For Users: Installation

### Quick Install (Recommended)
```bash
# Create environment
conda env create -f environment_user.yml
conda activate argprism

# Verify
argprism --version
```

### Manual Install
```bash
# Create environment
conda create -n argprism python=3.13 -y
conda activate argprism

# Install tools
conda install -c bioconda -c conda-forge diamond=2.1.9 blast=2.16.0 -y
conda install pytorch=2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install Python packages
pip install -r requirements.txt
```

## For Users: Running ARGprism

### Basic Command
```bash
argprism \
  -i input.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  -o results/
```

### Batch Processing
```bash
./run_batch.sh
```

## For Developers: Building Package

### Quick Build
```bash
# Prepare data
cp ARGPrismDB.fasta argprism/data/
cp metadata_arg.json argprism/data/
cp trained_model/best_model_fold4.pth argprism/models/

# Build
./build_package.sh

# Output: conda-build/noarch/argprism-1.0.0-py_0.tar.bz2
```

### Install Built Package
```bash
conda install conda-build/noarch/argprism-1.0.0-py_0.tar.bz2
```

## Key Files

| File | Purpose |
|------|---------|
| `environment_user.yml` | User environment specification |
| `setup.py` | Python package configuration |
| `meta.yaml` | Conda package recipe |
| `argprism/cli.py` | Command-line interface |
| `argprism/pipeline.py` | Main pipeline logic |
| `build_package.sh` | Automated build script |
| `run_batch.sh` | Batch processing script |

## Dependencies

### Conda-managed
- Python 3.13
- PyTorch 2.6.0 + CUDA 12.4
- DIAMOND 2.1.9
- BLAST 2.16.0

### Pip-managed
- transformers 4.49.0
- biopython 1.85
- numpy, pandas, scipy, scikit-learn
- matplotlib, seaborn
- h5py, tqdm, requests

## Important Notes

1. **GPU Support**: Automatic detection, use `--cpu` flag for CPU-only
2. **Input Format**: Protein sequences only (FASTA)
3. **Output**: CSV report with ARG predictions and annotations
4. **Performance**: GPU recommended for >1000 sequences

## Documentation

- **README.md**: Overview and quick start
- **INSTALL.md**: Detailed installation instructions
- **USER_GUIDE.md**: Complete usage guide
- **PACKAGE_BUILD_GUIDE.md**: Package creation guide

## Support

- GitHub: https://github.com/muneebdev7/ARGprism
- Issues: https://github.com/muneebdev7/ARGprism/issues

---
**Version**: 1.0.0 | **Updated**: November 2025
