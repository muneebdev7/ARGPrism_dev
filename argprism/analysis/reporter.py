"""Reporting output for ARGprism."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple


def load_metadata(metadata_json: Path) -> Dict[str, Dict[str, str]]:
    with metadata_json.open() as handle:
        return json.load(handle)


def parse_diamond_hits(diamond_tsv: Path) -> Dict[str, Tuple[str, float]]:
    best_hits: Dict[str, Tuple[str, float]] = {}
    if not diamond_tsv.exists():
        return best_hits
    with diamond_tsv.open() as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            qid, sid, _pid, _length, _evalue, bitscore = parts
            score = float(bitscore)
            if qid not in best_hits or score > best_hits[qid][1]:
                best_hits[qid] = (sid, score)
    return best_hits


def generate_report(
    predictions: Dict[str, str],
    best_hits: Dict[str, Tuple[str, float]],
    arg_metadata: Dict[str, Dict[str, str]],
    output_csv: Path,
) -> None:
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sequence_ID", "Predicted_Class", "ARG_Ref_ID", "ARG_Name", "Drug"])
        for seq_id, label in predictions.items():
            if label == "ARG":
                if seq_id in best_hits:
                    ref_id = best_hits[seq_id][0]
                    meta = arg_metadata.get(ref_id, {})
                    writer.writerow(
                        [
                            seq_id,
                            "ARG",
                            ref_id,
                            meta.get("ARG", ""),
                            meta.get("Drug", ""),
                        ]
                    )
                else:
                    writer.writerow([seq_id, "ARG", "", "", ""])
            else:
                writer.writerow([seq_id, "Non-ARG", "", "", ""])
