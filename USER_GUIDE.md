# ARGprism User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Input Preparation](#input-preparation)
3. [Running ARGprism](#running-argprism)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

---

## Getting Started

### Prerequisites Check

Before using ARGprism, ensure:

```bash
# Activate environment
conda activate argprism

# Check installation
argprism --version

# Check GPU availability (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check DIAMOND
diamond version
```

### Directory Structure

```
your_project/
├── input_sequences/       # Your protein FASTA files
├── results/              # Output directory
├── ARGPrismDB.fasta      # Reference database
├── metadata_arg.json     # ARG metadata
└── trained_model/
    └── best_model_fold4.pth
```

---

## Input Preparation

### 1. Protein Sequences

ARGprism expects **protein sequences** in FASTA format:

✅ **Correct Format:**
```
>sequence_001
MKIVKRILLVLLSLFFTVEYSNAQTDNLTLKIENVLKAKNARIGVAIFNSNEKDTLKI
NNDFHYPMQSVMKFPIALAVLSEIDKGNLSFEQKIEITPQDLLPKTWSPIKEEFPNGT
>sequence_002
MSLYRRLVLLSCLSWPLAGFSATALTNLVAEPFAKLEQDFGGSIGVYAMDTGSGATVS
```

❌ **Incorrect Format (DNA sequences):**
```
>sequence_001
ATGGCTAAGATCGTACGTCGTATTCTGCTGGTGTTGCTGTCGCTGTTTTTCTTCACG
```

### 2. Sequence Requirements

- **Format**: FASTA (`.faa`, `.fa`, `.fasta`)
- **Type**: Protein sequences only (amino acids)
- **Length**: Any length (typically 50-2000 amino acids)
- **Headers**: Must be unique
- **Characters**: Standard 20 amino acids (non-standard AAs replaced automatically)

### 3. Converting DNA to Protein

If you have DNA sequences, translate them first:

```bash
# Using Prodigal
prodigal -i input.fna -a output.faa -p meta

# Using transeq (EMBOSS)
transeq -sequence input.fna -outseq output.faa -frame 6
```

---

## Running ARGprism

### Basic Usage

```bash
argprism \
  -i input_sequences/my_proteins.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  -o results/
```

### Step-by-Step Example

#### Step 1: Prepare Your Data

```bash
# Create working directory
mkdir argprism_analysis
cd argprism_analysis

# Copy or link required files
ln -s ../ARGprism/trained_model .
ln -s ../ARGprism/ARGPrismDB.fasta .
ln -s ../ARGprism/metadata_arg.json .

# Place your input file
cp /path/to/your_proteins.faa input.faa
```

#### Step 2: Activate Environment

```bash
conda activate argprism
```

#### Step 3: Run ARGprism

```bash
argprism \
  -i input.faa \
  -c trained_model/best_model_fold4.pth \
  -d ARGPrismDB.fasta \
  -m metadata_arg.json \
  -o results/

# Watch the progress:
# Progress: 100.00% | Elapsed: 45.23s | Estimated Remaining: 0.00s
```

#### Step 4: Check Results

```bash
# View output files
ls -lh results/

# Quick summary
echo "Total ARGs predicted:"
grep -c "ARG" results/final_ARG_prediction_report.csv
```

---

## Understanding Results

### Output Files

1. **final_ARG_prediction_report.csv** - Main results file
2. **predicted_ARGs.fasta** - ARG sequences only
3. **predicted_ARGs_vs_ref.tsv** - BLAST alignment details
4. **diamond_arg_db.dmnd** - Database index (auto-generated)

### Reading the Report

**Example `final_ARG_prediction_report.csv`:**

```csv
Sequence_ID,Predicted_Class,ARG_Ref_ID,ARG_Name,Drug
seq_001,ARG,ARGPrism|WP_123456|...,TEM-1,beta_lactam
seq_002,Non-ARG,,,
seq_003,ARG,ARGPrism|WP_789012|...,NDM-1,carbapenem
seq_004,ARG,,,  # ARG but no database match
```

#### Column Descriptions:

| Column | Description | Values |
|--------|-------------|--------|
| Sequence_ID | Your sequence identifier | From input FASTA |
| Predicted_Class | ARG prediction | ARG or Non-ARG |
| ARG_Ref_ID | Best match in database | Empty if no match |
| ARG_Name | Gene/protein name | e.g., TEM-1, NDM-1 |
| Drug | Antibiotic class | e.g., beta_lactam |

### Interpreting Results

#### Case 1: Confident ARG Prediction
```csv
seq_001,ARG,ARGPrism|WP_123456|...,TEM-1,beta_lactam
```
- ✅ Predicted as ARG
- ✅ Has database match
- ✅ Known ARG name and drug class
- **Interpretation**: High-confidence ARG, confers beta-lactam resistance

#### Case 2: Novel ARG Candidate
```csv
seq_004,ARG,,,
```
- ✅ Predicted as ARG
- ❌ No database match
- **Interpretation**: Potentially novel ARG, requires manual verification

#### Case 3: Non-ARG
```csv
seq_002,Non-ARG,,,
```
- ❌ Not predicted as ARG
- **Interpretation**: Likely not an antibiotic resistance protein

---

## Advanced Usage

### 1. Batch Processing

Process multiple samples efficiently:

```bash
# Use the provided script
./run_batch.sh

# Or manually
for faa in input_sequences/*.faa; do
  sample=$(basename "$faa" .faa)
  argprism -i "$faa" -c model.pth -d db.fasta -m meta.json -o "results_${sample}/"
done
```

### 2. CPU-Only Mode

For systems without GPU:

```bash
argprism -i input.faa ... --cpu
```

### 3. Using Python API

For integration into larger pipelines:

```python
from argprism import ARGPrismPipeline

# Initialize
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

### 4. Custom Analysis

```python
import pandas as pd

# Load results
df = pd.read_csv("results/final_ARG_prediction_report.csv")

# Count ARGs by drug class
drug_counts = df[df['Predicted_Class'] == 'ARG']['Drug'].value_counts()
print(drug_counts)

# Get novel ARG candidates
novel_args = df[
    (df['Predicted_Class'] == 'ARG') & 
    (df['ARG_Ref_ID'] == '')
]
print(f"Novel ARG candidates: {len(novel_args)}")
```

---

## Best Practices

### 1. Data Quality

✅ **DO:**
- Use high-quality protein sequences
- Remove very short sequences (<30 aa)
- Use complete or nearly complete protein sequences
- Check for frame shifts in translated sequences

❌ **DON'T:**
- Use DNA/RNA sequences directly
- Include sequences with many ambiguous amino acids (X)
- Mix protein and DNA sequences

### 2. Resource Management

#### For Large Datasets (>10,000 sequences):

```bash
# Split into batches
split -l 4000 large_input.faa batch_

# Process batches
for batch in batch_*; do
  argprism -i "$batch" -c model.pth -d db.fasta -m meta.json -o "results_${batch}/"
done

# Merge results
cat results_batch_*/final_ARG_prediction_report.csv > combined_results.csv
```

#### For GPU Memory Issues:

```bash
# Use CPU instead
argprism -i input.faa ... --cpu

# Or process smaller batches
```

### 3. Result Validation

Always validate important findings:

1. **Check alignment quality** in `predicted_ARGs_vs_ref.tsv`
2. **Verify novel ARGs** with additional tools (e.g., BLAST, HMM)
3. **Compare with other ARG databases** (CARD, ResFinder)

---

## FAQ

### Q: How long does it take to process 1,000 sequences?

**A:** Approximately:
- GPU (NVIDIA V100): ~3 minutes
- CPU (8 cores): ~40 minutes

Time scales roughly linearly with sequence count.

### Q: Can I use my own trained model?

**A:** Yes! Just provide your `.pth` file:

```bash
argprism -i input.faa -c my_custom_model.pth -d db.fasta -m meta.json
```

The model must have the same architecture as `ARGClassifier`.

### Q: What if DIAMOND fails?

**A:** Common solutions:

```bash
# Rebuild database
rm diamond_arg_db.dmnd
argprism -i input.faa ...

# Check DIAMOND installation
diamond version

# Reinstall if needed
conda install -c bioconda diamond=2.1.9 -y
```

### Q: Can I add my own ARG database?

**A:** Yes! Format requirements:

1. **FASTA format** with unique headers
2. **Metadata JSON** matching your headers:

```json
{
  "your_header_id": {
    "ARG_name": "gene_name",
    "drug": "antibiotic_class"
  }
}
```

### Q: How to interpret sequences with no database match?

**A:** These could be:
1. **Novel ARGs** - Not yet in databases
2. **Distant homologs** - Below similarity threshold
3. **False positives** - Check with additional tools

Recommended validation:
```bash
# BLAST against nr database
blastp -query novel_arg.faa -db nr -remote -outfmt 6

# Check conserved domains
rpsblast -query novel_arg.faa -db Cdd -outfmt 6
```

### Q: Can I analyze metagenome-assembled genomes (MAGs)?

**A:** Yes! Workflow:

```bash
# 1. Predict genes
prodigal -i MAG.fna -a MAG_proteins.faa -p meta

# 2. Run ARGprism
argprism -i MAG_proteins.faa -c model.pth -d db.fasta -m meta.json -o results/
```

### Q: What's the minimum sequence length?

**A:** ARGprism works on any length, but:
- **Recommended**: >50 amino acids
- **Minimum**: >30 amino acids
- Very short sequences (<30 aa) may give unreliable predictions

### Q: How to cite ARGprism?

**A:** See README.md for citation information.

---

## Getting Help

### Still have questions?

1. **Check documentation**: README.md, INSTALL.md
2. **Search existing issues**: [GitHub Issues](https://github.com/muneebdev7/ARGprism/issues)
3. **Open new issue**: Include:
   - OS and Python version
   - Complete error message
   - Steps to reproduce
   - Sample data (if possible)

### Useful Commands for Bug Reports

```bash
# System info
uname -a
python --version
conda --version

# Environment info
conda list

# GPU info (if applicable)
nvidia-smi

# ARGprism version
argprism --version
```

---

**Happy ARG hunting! 🧬🔬**
