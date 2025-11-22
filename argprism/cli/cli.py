"""Command line interface for ARGprism."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

# Set threading environment variables BEFORE any PyTorch/ML imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from .. import __version__
from ..io.file_paths import (
    DEFAULT_DIAMOND_BLAST_OUTPUT_FILENAME,
    DEFAULT_PREDICTED_ARGS_FASTA_FILENAME,
    DEFAULT_FINAL_REPORT_CSV_FILENAME,
)
from ..core.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argprism",
        description="Deep learning-based antibiotic resistance gene prediction pipeline.",
    )
    parser.add_argument(
        "input_fasta",
        help="Input protein FASTA file containing sequences to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        default="argprism_output",
        help="Directory where pipeline outputs will be written (default: argprism_output).",
    )
    parser.add_argument(
        "--output-fasta",
        default=DEFAULT_PREDICTED_ARGS_FASTA_FILENAME,
        help="Filename for predicted ARG sequences (default: predicted_ARGs.fasta).",
    )
    parser.add_argument(
        "--diamond-output",
        default=DEFAULT_DIAMOND_BLAST_OUTPUT_FILENAME,
        help="Filename for DIAMOND BLAST results (default: predicted_ARGs_vs_ref.tsv).",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_FINAL_REPORT_CSV_FILENAME,
        help="Filename for the final CSV report (default: final_ARG_prediction_report.csv).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Force execution on CPU or CUDA. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--reuse-diamond-db",
        action="store_true",
        help="Skip rebuilding the DIAMOND database if it already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"argprism {__version__}",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_pipeline(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        output_fasta=args.output_fasta,
        diamond_output=args.diamond_output,
        final_report=args.report,
        preferred_device=args.device,
        build_diamond_db=not args.reuse_diamond_db,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f"Predictions saved to {result.predicted_fasta}")
        if result.diamond_output:
            print(f"DIAMOND hits written to {result.diamond_output}")
        if result.report_csv:
            print(f"Final report available at {result.report_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
