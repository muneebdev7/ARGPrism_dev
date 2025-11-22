"""DIAMOND integration helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def run_diamond(
    predicted_fasta: Path,
    arg_db_fasta: Path,
    diamond_db_prefix: Path,
    diamond_output: Path,
    diamond_executable: str = "diamond",
    build_db: bool = True,
    verbose: bool = True,
) -> None:
    db_path = diamond_db_prefix.with_suffix(".dmnd")

    if build_db and not db_path.exists():
        cmd_makedb = [diamond_executable, "makedb", "--in", str(arg_db_fasta), "-d", str(diamond_db_prefix)]
        if verbose:
            print("Running:", " ".join(cmd_makedb))
        subprocess.run(cmd_makedb, check=True)

    cmd_blastp = [
        diamond_executable,
        "blastp",
        "-q",
        str(predicted_fasta),
        "-d",
        str(diamond_db_prefix),
        "-o",
        str(diamond_output),
        "-f",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "evalue",
        "bitscore",
    ]
    if verbose:
        print("Running:", " ".join(cmd_blastp))
    subprocess.run(cmd_blastp, check=True)
