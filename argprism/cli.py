#!/usr/bin/env python
"""
Command-line interface for ARGprism
"""

import argparse
import sys
import os
from pathlib import Path

from .pipeline import ARGPrismPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ARGprism: Deep Learning-based Antibiotic Resistance Gene Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  argprism -i input.faa -c model.pth -d ARGPrismDB.fasta -m metadata.json
  
  # Specify output directory
  argprism -i input.faa -c model.pth -d ARGPrismDB.fasta -m metadata.json -o results/
  
  # Force CPU usage
  argprism -i input.faa -c model.pth -d ARGPrismDB.fasta -m metadata.json --cpu
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input protein sequences in FASTA format'
    )
    
    parser.add_argument(
        '-c', '--classifier',
        required=True,
        help='Path to trained ARG classifier model (.pth file)'
    )
    
    parser.add_argument(
        '-d', '--database',
        required=True,
        help='ARG reference database in FASTA format'
    )
    
    parser.add_argument(
        '-m', '--metadata',
        required=True,
        help='ARG metadata in JSON format'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (default: auto-detect GPU)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='ARGprism 1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input files
    for filepath, name in [
        (args.input, 'Input file'),
        (args.classifier, 'Classifier model'),
        (args.database, 'Database file'),
        (args.metadata, 'Metadata file')
    ]:
        if not os.path.exists(filepath):
            print(f"Error: {name} not found: {filepath}", file=sys.stderr)
            sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    # Determine device
    device = 'cpu' if args.cpu else None
    
    # Run pipeline
    try:
        print("="*60)
        print("ARGprism: Antibiotic Resistance Gene Prediction Pipeline")
        print("="*60)
        
        pipeline = ARGPrismPipeline(args.classifier, device=device)
        pipeline.run(
            input_fasta=args.input,
            arg_db_fasta=args.database,
            metadata_json=args.metadata,
            output_dir=args.output
        )
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
