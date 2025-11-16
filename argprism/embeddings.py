"""ProtAlbert embedding utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from transformers import AlbertModel, AlbertTokenizer

from .constants import PROT_ALBERT_MODEL


def load_plm(device: torch.device) -> tuple[AlbertTokenizer, AlbertModel]:
    """Load the ProtAlbert tokenizer and model on the requested device."""
    tokenizer = AlbertTokenizer.from_pretrained(PROT_ALBERT_MODEL, do_lower_case=False)
    model = AlbertModel.from_pretrained(PROT_ALBERT_MODEL).to(device)
    model.eval()
    return tokenizer, model


def _clean_sequence(sequence: str) -> str:
    return sequence.replace("U", "X").replace("Z", "X").replace("O", "X")


def embed_sequence(sequence: str, tokenizer: AlbertTokenizer, model: AlbertModel, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(" ".join(sequence), return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()


def generate_embeddings(
    input_fasta: Path,
    tokenizer: AlbertTokenizer,
    model: AlbertModel,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, SeqRecord]]:
    embeddings: Dict[str, torch.Tensor] = {}
    sequences: Dict[str, SeqRecord] = {}
    records = list(SeqIO.parse(input_fasta, "fasta"))
    total = len(records)
    start_time = time.time()

    for index, record in enumerate(records, start=1):
        seq = _clean_sequence(str(record.seq))
        sequences[record.id] = record
        embeddings[record.id] = embed_sequence(seq, tokenizer, model, device)

        if verbose:
            elapsed = time.time() - start_time
            progress = index / total if total else 1.0
            remaining = (elapsed / progress) - elapsed if progress else 0.0
            print(
                f"\rEmbedding {index}/{total}"
                f" | {progress*100:.2f}%"
                f" | Elapsed {elapsed:.1f}s"
                f" | Remaining {remaining:.1f}s",
                end="",
            )
    if verbose:
        print()
    return embeddings, sequences
