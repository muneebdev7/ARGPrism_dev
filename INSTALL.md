# ARGprism Installation Guide

## Quick Start

### Option 1: Install via Conda (Recommended)

This is the easiest and most reliable installation method.

```bash
# Step 1: Create a new conda environment from the environment file
conda env create -f environment_user.yml

# Step 2: Activate the environment
conda activate argprism

# Step 3: Verify installation
argprism --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
diamond version
```

### Option 2: Manual Installation

If you prefer to set up the environment manually:

```bash
# Step 1: Create a new conda environment with Python 3.13
conda create -n argprism python=3.13 -y

# Step 2: Activate the environment
conda activate argprism

# Step 3: Install bioinformatics tools via conda
conda install -c bioconda -c conda-forge diamond=2.1.9 blast=2.16.0 -y

# Step 4: Install PyTorch with CUDA support
conda install pytorch=2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Step 5: Install Python dependencies via pip
pip install -r argprism_requirements.txt

# Step 6: Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
diamond version
```

---

## System Requirements

### Minimum Requirements:
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+, or similar)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.11, 3.12, or 3.13

### Recommended Requirements:
- **OS**: Linux (Ubuntu 22.04+ or CentOS 8+)
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **GPU**: NVIDIA GPU with CUDA 12.4+ support (for faster processing)
- **Storage**: 20+ GB free space
- **Python**: 3.13

---

## GPU Support

ARGprism can leverage NVIDIA GPUs to significantly speed up embedding generation.

### Check GPU Availability:
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU information
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### CPU-Only Installation:
If you don't have a GPU or want to use CPU only:

```bash
# Install PyTorch CPU version
conda install pytorch=2.6.0 cpuonly -c pytorch -y
```

---

## Dependency Details

### Core Dependencies:

1. **Python Packages** (installed via pip):
   - `torch==2.6.0` - PyTorch deep learning framework
   - `transformers==4.49.0` - Hugging Face transformers (for ProtAlbert)
   - `biopython==1.85` - Biological sequence analysis
   - `numpy==2.2.3` - Numerical computing
   - `pandas==2.2.3` - Data manipulation
   - `scikit-learn==1.6.1` - Machine learning utilities
   - `h5py==3.13.0` - HDF5 file format support

2. **Bioinformatics Tools** (installed via conda):
   - `diamond==2.1.9` - Fast sequence alignment tool
   - `blast==2.16.0` - NCBI BLAST tools

3. **CUDA Libraries** (automatically installed with PyTorch):
   - CUDA Toolkit 12.4
   - cuDNN 9.1.0

---

## Troubleshooting

### Issue 1: CUDA version mismatch
**Error**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Solution**: Ensure your NVIDIA driver supports CUDA 12.4+
```bash
nvidia-smi  # Check driver version
```

If your driver is older, either:
- Update your NVIDIA driver, or
- Install CPU-only version of PyTorch

### Issue 2: Out of memory (GPU)
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: 
```bash
# Use CPU instead
argprism -i input.faa -c model.pth -d db.fasta -m metadata.json --cpu
```

### Issue 3: DIAMOND not found
**Error**: `diamond: command not found`

**Solution**: 
```bash
# Reinstall DIAMOND
conda install -c bioconda diamond=2.1.9 -y
```

### Issue 4: Import errors
**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**: 
```bash
# Reinstall Python dependencies
pip install -r argprism_requirements.txt
```

---

## Verifying Installation

Run the following commands to verify your installation:

```bash
# Activate environment
conda activate argprism

# Check Python version
python --version

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check Biopython
python -c "from Bio import SeqIO; print('Biopython: OK')"

# Check DIAMOND
diamond version

# Check BLAST
blastp -version
```

All commands should execute without errors.

---

## Updating ARGprism

To update ARGprism to the latest version:

```bash
# Activate environment
conda activate argprism

# Update conda packages
conda update --all -y

# Update pip packages
pip install --upgrade -r argprism_requirements.txt
```

---

## Uninstallation

To completely remove ARGprism:

```bash
# Remove the conda environment
conda env remove -n argprism

# Clean conda cache (optional)
conda clean --all -y
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/muneebdev7/ARGprism/issues)
2. Open a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

## Additional Notes

### For HPC/Cluster Users:
If you're installing on an HPC system:

```bash
# Load required modules (example for SLURM)
module load cuda/12.4
module load gcc/11.2.0

# Then proceed with conda installation
conda env create -f environment_user.yml
```

### For Docker Users:
A Dockerfile will be provided in future releases for containerized deployment.

---

**Last Updated**: November 2025
**ARGprism Version**: 1.0.0
