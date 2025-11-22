"""Core pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..classifier.arg_classifier import ARGClassifier, classify_embeddings, load_classifier
from ..io.file_paths import (
    DEFAULT_ARG_DB,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DIAMOND_OUTPUT,
    DEFAULT_DIAMOND_PREFIX,
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_FASTA,
    DEFAULT_REPORT,
    PACKAGE_ROOT,
)
from .device_manager import select_device
from ..analysis.diamond_mapping import run_diamond
from ..classifier.embeddings_generator import generate_embeddings, load_plm
from ..io.sequence_io import save_predicted_args
from ..analysis.reporter import generate_report, load_metadata, parse_diamond_hits


@dataclass
class PipelineResult:
    predictions: Dict[str, str]
    predicted_fasta: Path
    report_csv: Optional[Path]
    diamond_output: Optional[Path]
    elapsed_seconds: float


def run_pipeline(
    input_fasta: Path | str,
    output_dir: Path | str,
    classifier_path: Path | str = DEFAULT_CLASSIFIER_PATH,
    arg_db_fasta: Path | str = DEFAULT_ARG_DB,
    metadata_json: Path | str = DEFAULT_METADATA,
    output_fasta: Path | str = DEFAULT_OUTPUT_FASTA,
    diamond_db_prefix: Path | str = DEFAULT_DIAMOND_PREFIX,
    diamond_output: Path | str = DEFAULT_DIAMOND_OUTPUT,
    final_report: Path | str = DEFAULT_REPORT,
    preferred_device: Optional[str] = None,
    diamond_executable: str = "diamond",
    build_diamond_db: bool = True,
    verbose: bool = True,
) -> PipelineResult:
    start_time = time.time()

    input_fasta = Path(input_fasta)
    output_dir = Path(output_dir)
    classifier_path = Path(classifier_path)
    arg_db_fasta = Path(arg_db_fasta)
    metadata_json = Path(metadata_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_fasta = output_dir / Path(output_fasta)
    diamond_db_prefix = output_dir / Path(diamond_db_prefix)
    diamond_output = output_dir / Path(diamond_output)
    final_report_path = output_dir / Path(final_report)

    if verbose:
        print(f"Loading resources from {PACKAGE_ROOT}")

    device = select_device(preferred_device)
    if verbose:
        print(f"Using device: {device}")

    tokenizer, model = load_plm(device)
    if verbose:
        print("Generating embeddings for input sequences...")
    embeddings, sequences = generate_embeddings(input_fasta, tokenizer, model, device, verbose=verbose)

    if verbose:
        print("Loading classifier...")
    classifier = load_classifier(classifier_path, device)

    if verbose:
        print("Classifying sequences...")
    predictions = classify_embeddings(embeddings, classifier, device)

    if verbose:
        print("Saving predicted ARG sequences...")
    predicted_count = save_predicted_args(sequences, predictions, output_fasta)

    diamond_path: Optional[Path] = None
    report_path: Optional[Path] = None

    if predicted_count and verbose:
        print(f"Saved {predicted_count} predicted ARG sequences to {output_fasta}")

    if predicted_count:
        if verbose:
            print("Mapping predicted ARGs to reference ARG database with DIAMOND...")
        run_diamond(
            output_fasta,
            arg_db_fasta,
            diamond_db_prefix,
            diamond_output,
            diamond_executable=diamond_executable,
            build_db=build_diamond_db,
            verbose=verbose,
        )
        diamond_path = diamond_output

        if verbose:
            print("Loading ARG metadata...")
        metadata = load_metadata(metadata_json)

        if verbose:
            print("Parsing DIAMOND mapping results...")
        best_hits = parse_diamond_hits(diamond_output)

        if verbose:
            print("Generating final annotated report...")
        generate_report(predictions, best_hits, metadata, final_report_path)
        report_path = final_report_path
    elif verbose:
        print("No ARG predictions were made; skipping DIAMOND mapping and report generation.")

    elapsed = time.time() - start_time
    if verbose:
        print(f"Pipeline complete in {elapsed:.2f} seconds")

    return PipelineResult(
        predictions=predictions,
        predicted_fasta=output_fasta,
        report_csv=report_path,
        diamond_output=diamond_path,
        elapsed_seconds=elapsed,
    )
