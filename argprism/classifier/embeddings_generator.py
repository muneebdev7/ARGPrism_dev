"""ProtAlbert embedding utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from transformers import AlbertModel, AlbertTokenizer

from ..io.file_paths import PROT_ALBERT_MODEL


def load_plm(device: torch.device) -> tuple[AlbertTokenizer, AlbertModel]:
    """Load the ProtAlbert tokenizer and model on the requested device."""
    tokenizer = AlbertTokenizer.from_pretrained(PROT_ALBERT_MODEL, do_lower_case=False)
    model = AlbertModel.from_pretrained(PROT_ALBERT_MODEL).to(device)
    model.eval()
    return tokenizer, model


def _clean_sequence(sequence: str) -> str:
    return sequence.replace("U", "X").replace("Z", "X").replace("O", "X")


def _get_optimal_batch_size(device: torch.device, sequence_count: int) -> int:
    """Determine optimal batch size based on device and sequence count."""
    if device.type == "cuda":
        # For GPU, use larger batches but be memory conscious
        return min(1, sequence_count)
    else:
        # For CPU, use smaller batches to balance memory and speed
        return min(16, sequence_count)


def embed_sequences_batch(
    sequences: List[str], 
    tokenizer: AlbertTokenizer, 
    model: AlbertModel, 
    device: torch.device,
    max_length: int = 512
) -> torch.Tensor:
    """Embed a batch of sequences efficiently."""
    # Prepare sequences for tokenization
    spaced_sequences = [" ".join(seq) for seq in sequences]
    
    # Tokenize all sequences in the batch
    tokens = tokenizer(
        spaced_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate embeddings for the entire batch
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Return mean-pooled embeddings for each sequence
    return outputs.last_hidden_state.mean(dim=1).cpu()


def generate_embeddings(
    input_fasta: Path,
    tokenizer: AlbertTokenizer,
    model: AlbertModel,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, SeqRecord]]:
    """Generate embeddings for all sequences using efficient batch processing."""
    sequences: Dict[str, SeqRecord] = {}
    records = list(SeqIO.parse(input_fasta, "fasta"))
    total = len(records)
    
    if total == 0:
        return {}, {}
    
    # Prepare data
    sequence_ids = []
    sequence_strings = []
    
    for record in records:
        cleaned_seq = _clean_sequence(str(record.seq))
        sequences[record.id] = record
        sequence_ids.append(record.id)
        sequence_strings.append(cleaned_seq)
    
    # Determine optimal batch size
    batch_size = _get_optimal_batch_size(device, total)
    
    embeddings: Dict[str, torch.Tensor] = {}
    start_time = time.time()
    
    if verbose:
        print(f"Processing {total} sequences in a batch size of {batch_size} sequences...")
    
    # Process sequences in batches
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_ids = sequence_ids[i:batch_end]
        batch_sequences = sequence_strings[i:batch_end]
        
        # Generate embeddings for the batch
        batch_embeddings = embed_sequences_batch(
            batch_sequences, tokenizer, model, device
        )
        
        # Store embeddings with their corresponding IDs
        for j, seq_id in enumerate(batch_ids):
            embeddings[seq_id] = batch_embeddings[j]
        
        if verbose:
            processed = batch_end
            elapsed = time.time() - start_time
            progress = processed / total
            remaining = (elapsed / progress) - elapsed if progress > 0 else 0.0
            
            print(
                f"\rEmbedding batch {(i // batch_size) + 1}/{(total + batch_size - 1) // batch_size} "
                f"| {processed}/{total} ({progress*100:.1f}%) "
                f"| Elapsed {elapsed:.1f}s | Remaining {remaining:.1f}s",
                end="",
            )
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\nCompleted embedding generation in {elapsed:.1f}s")
        print(f"Average time per sequence: {elapsed/total:.3f}s")
    
    return embeddings, sequences


# Keep the single sequence function for backwards compatibility
def embed_sequence(sequence: str, tokenizer: AlbertTokenizer, model: AlbertModel, device: torch.device) -> torch.Tensor:
    """Embed a single sequence (legacy function for compatibility)."""
    return embed_sequences_batch([sequence], tokenizer, model, device)[0]
