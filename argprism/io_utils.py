"""Input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def save_predicted_args(
    sequences: Dict[str, SeqRecord],
    predictions: Dict[str, str],
    output_fasta: Path,
) -> int:
    arg_records = [sequences[seq_id] for seq_id, label in predictions.items() if label == "ARG" and seq_id in sequences]
    count = SeqIO.write(arg_records, output_fasta, "fasta")
    return count
