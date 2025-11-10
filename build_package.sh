#!/bin/bash
# Script to build ARGprism conda package

set -e  # Exit on error

echo "======================================"
echo "Building ARGprism Conda Package"
echo "======================================"

# Check if conda-build is installed
if ! command -v conda-build &> /dev/null
then
    echo "conda-build not found. Installing..."
    conda install conda-build -y
fi

# Create build directory
mkdir -p conda-build

# Copy required files to argprism package structure
echo "Setting up package structure..."
cp ARGPrismDB.fasta argprism/data/
cp metadata_arg.json argprism/data/
cp trained_model/best_model_fold4.pth argprism/models/

# Build the package
echo "Building conda package..."
conda-build . --output-folder conda-build

# Get the package path
PACKAGE=$(conda-build . --output)

echo ""
echo "======================================"
echo "Build completed successfully!"
echo "Package location: $PACKAGE"
echo "======================================"
echo ""
echo "To install the package locally:"
echo "  conda install $PACKAGE"
echo ""
echo "To upload to anaconda.org:"
echo "  anaconda upload $PACKAGE"
echo ""
