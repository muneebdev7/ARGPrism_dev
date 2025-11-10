#!/bin/bash
# Example script to run ARGprism on all input samples

set -e  # Exit on error

# Configuration
CLASSIFIER="models/best_model_fold4.pth"
DATABASE="ARGPrismDB.fasta"
METADATA="metadata_arg.json"
INPUT_DIR="Input_proteins"
OUTPUT_BASE="batch_results"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "argprism" ]; then
    echo "Error: Please activate the argprism conda environment first:"
    echo "  conda activate argprism"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Count total samples
total_samples=$(ls "$INPUT_DIR"/*.faa 2>/dev/null | wc -l)

if [ "$total_samples" -eq 0 ]; then
    echo "Error: No .faa files found in $INPUT_DIR"
    exit 1
fi

echo "======================================"
echo "ARGprism Batch Processing"
echo "======================================"
echo "Total samples: $total_samples"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_BASE"
echo "======================================"
echo ""

# Process each sample
current=0
for faa in "$INPUT_DIR"/*.faa; do
    current=$((current + 1))
    sample=$(basename "$faa" _PROT.faa)
    output_dir="$OUTPUT_BASE/$sample"
    
    echo "[$current/$total_samples] Processing: $sample"
    echo "  Input: $faa"
    echo "  Output: $output_dir"
    
    # Create sample-specific output directory
    mkdir -p "$output_dir"
    
    # Run ARGprism
    argprism \
        -i "$faa" \
        -c "$CLASSIFIER" \
        -d "$DATABASE" \
        -m "$METADATA" \
        -o "$output_dir"
    
    echo "  ✓ Completed: $sample"
    echo ""
done

echo "======================================"
echo "Batch processing completed!"
echo "Results saved in: $OUTPUT_BASE"
echo "======================================"

# Generate summary report
echo ""
echo "Generating summary report..."
python3 << 'EOF'
import os
import csv
import pandas as pd

output_base = "batch_results"
samples = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]

summary = []
for sample in samples:
    report_path = os.path.join(output_base, sample, "final_ARG_prediction_report.csv")
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        total = len(df)
        n_args = len(df[df['Predicted_Class'] == 'ARG'])
        summary.append({
            'Sample': sample,
            'Total_Sequences': total,
            'Predicted_ARGs': n_args,
            'ARG_Percentage': f"{(n_args/total*100):.2f}%"
        })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values('Sample')

# Save summary
summary_path = os.path.join(output_base, "batch_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\nSummary report saved to: {summary_path}")
print("\n" + "="*60)
print("BATCH SUMMARY")
print("="*60)
print(summary_df.to_string(index=False))
print("="*60)
EOF

echo ""
echo "Done!"
