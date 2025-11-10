# ARGprism Conda Package Creation Guide

This document explains how to create a distributable conda package for ARGprism.

## Package Structure

```
ARGprism/
├── argprism/                    # Main Python package
│   ├── __init__.py             # Package initialization
│   ├── classifier.py           # Neural network classifier
│   ├── embeddings.py           # ProtAlbert embedding generation
│   ├── pipeline.py             # Main pipeline logic
│   ├── cli.py                  # Command-line interface
│   ├── data/                   # Data files
│   │   ├── ARGPrismDB.fasta    # Reference database
│   │   └── metadata_arg.json   # ARG metadata
│   └── models/                 # Model files
│       └── best_model_fold4.pth # Trained model
├── setup.py                    # Python package setup
├── meta.yaml                   # Conda package recipe
├── environment_user.yml        # User environment specification
├── requirements.txt            # Python dependencies
├── MANIFEST.in                 # Package data manifest
├── README.md                   # Main documentation
├── INSTALL.md                  # Installation guide
├── USER_GUIDE.md              # User guide
├── LICENSE                     # MIT License
├── build_package.sh           # Build script
└── run_batch.sh               # Batch processing script
```

## Building the Conda Package

### Prerequisites

1. **Install conda-build**:
   ```bash
   conda install conda-build anaconda-client -y
   ```

2. **Prepare package data**:
   ```bash
   # Copy data files to package directories
   cp ARGPrismDB.fasta argprism/data/
   cp metadata_arg.json argprism/data/
   cp trained_model/best_model_fold4.pth argprism/models/
   ```

### Build Process

#### Method 1: Using the Build Script (Recommended)

```bash
# Run the automated build script
./build_package.sh
```

This script will:
1. Check for conda-build
2. Copy necessary data files
3. Build the conda package
4. Display the package location

#### Method 2: Manual Build

```bash
# Build the package
conda-build . --output-folder conda-build

# The package will be created in conda-build/noarch/
```

### Package Output

After building, you'll get:
```
conda-build/
└── noarch/
    └── argprism-1.0.0-py_0.tar.bz2
```

## Installing the Package

### Local Installation

```bash
# Install from local build
conda install conda-build/noarch/argprism-1.0.0-py_0.tar.bz2
```

### Creating User Environment

Users can install ARGprism using the provided environment file:

```bash
# Create environment from file
conda env create -f environment_user.yml

# Activate environment
conda activate argprism

# Verify installation
argprism --version
```

## Publishing the Package

### To Anaconda.org (Conda-Forge)

1. **Create an account** on https://anaconda.org

2. **Login**:
   ```bash
   anaconda login
   ```

3. **Upload package**:
   ```bash
   anaconda upload conda-build/noarch/argprism-1.0.0-py_0.tar.bz2
   ```

4. **Users can install with**:
   ```bash
   conda install -c your-channel argprism
   ```

### To PyPI (for pip installation)

1. **Install build tools**:
   ```bash
   pip install build twine
   ```

2. **Build distribution**:
   ```bash
   python -m build
   ```

3. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

4. **Users can install with**:
   ```bash
   pip install argprism
   ```

## Version Management

To release a new version:

1. **Update version numbers** in:
   - `setup.py` → `version='1.x.x'`
   - `meta.yaml` → `{% set version = "1.x.x" %}`
   - `argprism/__init__.py` → `__version__ = "1.x.x"`

2. **Rebuild package**:
   ```bash
   ./build_package.sh
   ```

3. **Upload new version**:
   ```bash
   anaconda upload conda-build/noarch/argprism-1.x.x-py_0.tar.bz2
   ```

## Testing the Package

Before distributing, test the package:

### 1. Create Test Environment

```bash
# Create fresh environment
conda create -n argprism-test python=3.13 -y
conda activate argprism-test

# Install your built package
conda install conda-build/noarch/argprism-1.0.0-py_0.tar.bz2

# Install additional dependencies
conda install -c bioconda diamond=2.1.9 -y
```

### 2. Run Tests

```bash
# Test CLI
argprism --version

# Test imports
python -c "from argprism import ARGPrismPipeline; print('OK')"

# Test with sample data
argprism \
  -i Input_proteins/ERR589503_PROT.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  -o test_results/
```

### 3. Clean Up

```bash
conda deactivate
conda env remove -n argprism-test
```

## Dependencies Management

### Conda vs Pip Dependencies

ARGprism uses a hybrid approach:

**Conda-managed** (system-level, better compatibility):
- Python
- DIAMOND
- BLAST
- PyTorch (with CUDA)

**Pip-managed** (Python packages):
- NumPy, Pandas, SciPy
- Biopython
- Transformers
- Other Python libraries

This ensures:
1. ✅ Reliable PyTorch + CUDA installation
2. ✅ Bioinformatics tools work correctly
3. ✅ Easy Python dependency management

## Package Size Optimization

The trained model file is large (~80 MB). Options to reduce package size:

### Option 1: Separate Model Download

```python
# In argprism/cli.py, download model on first use
def download_model_if_needed(model_path):
    if not os.path.exists(model_path):
        url = "https://example.com/models/best_model_fold4.pth"
        download_file(url, model_path)
```

### Option 2: Use Model Registry

Upload model to Hugging Face Model Hub:
```python
from transformers import AutoModel

# Upload once
model.save_pretrained("your-username/argprism-classifier")

# Users download automatically
model = AutoModel.from_pretrained("your-username/argprism-classifier")
```

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/build-package.yml`:

```yaml
name: Build Conda Package

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - uses: conda-incubator/setup-miniconda@v2
      with:
        auto-update-conda: true
        
    - name: Build package
      run: |
        conda install conda-build -y
        ./build_package.sh
        
    - name: Upload to Anaconda
      env:
        ANACONDA_TOKEN: ${{ secrets.ANACONDA_TOKEN }}
      run: |
        conda install anaconda-client -y
        anaconda -t $ANACONDA_TOKEN upload conda-build/noarch/*.tar.bz2
```

## Troubleshooting Build Issues

### Issue 1: Missing files error

**Error**: `FileNotFoundError: ARGPrismDB.fasta`

**Solution**: Ensure data files are copied:
```bash
cp ARGPrismDB.fasta argprism/data/
cp metadata_arg.json argprism/data/
cp trained_model/best_model_fold4.pth argprism/models/
```

### Issue 2: Dependency conflicts

**Error**: `ResolvePackageNotFound: diamond=2.1.9`

**Solution**: Update channel priorities in `meta.yaml`:
```yaml
channels:
  - bioconda
  - conda-forge
  - defaults
```

### Issue 3: PyTorch CUDA version mismatch

**Solution**: Specify exact versions in `environment_user.yml`:
```yaml
- pytorch=2.6.0
- pytorch-cuda=12.4
```

## Platform-Specific Builds

### Linux

```bash
conda-build . --output-folder conda-build/linux-64
```

### macOS (CPU-only)

Modify `meta.yaml`:
```yaml
requirements:
  run:
    - pytorch=2.6.0
    - cpuonly  # No CUDA on macOS
```

### Windows

Windows support requires:
1. Windows-compatible DIAMOND alternative, or
2. WSL2 installation instructions

## Documentation Updates

When releasing, ensure:

- [ ] README.md is up to date
- [ ] INSTALL.md has correct commands
- [ ] USER_GUIDE.md reflects any changes
- [ ] Version numbers are consistent
- [ ] CHANGELOG is updated

## Release Checklist

Before releasing a new version:

- [ ] Update version numbers (3 files)
- [ ] Test package build
- [ ] Test package installation
- [ ] Run test suite
- [ ] Update documentation
- [ ] Create git tag
- [ ] Build package
- [ ] Upload to Anaconda/PyPI
- [ ] Create GitHub release
- [ ] Announce release

---

**Maintained by**: Muneeb  
**Last Updated**: November 2025  
**ARGprism Version**: 1.0.0
