"""
Protein embedding generation using ProtAlbert
"""

import torch
from Bio import SeqIO
from transformers import AlbertTokenizer, AlbertModel
import time


def load_protalbert_model(device='cuda'):
    """
    Load ProtAlbert model and tokenizer.
    
    Args:
        device: 'cuda' or 'cpu'
        
    Returns:
        tuple: (model, tokenizer)
    """
    model = AlbertModel.from_pretrained("Rostlab/prot_albert").to(device)
    tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
    return model, tokenizer


def generate_embedding(sequence, model, tokenizer, device='cuda'):
    """
    Generate embedding for a single protein sequence.
    
    Args:
        sequence: Protein sequence string
        model: ProtAlbert model
        tokenizer: ProtAlbert tokenizer
        device: 'cuda' or 'cpu'
        
    Returns:
        numpy array: 4096-dimensional embedding
    """
    # Preprocess sequence: replace non-standard amino acids
    seq = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    
    # Tokenize with spaces between amino acids
    tokens = tokenizer(' '.join(list(seq)), return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Generate embedding
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Use mean pooling across sequence length
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]


def generate_embeddings(input_fasta, model, tokenizer, device='cuda', verbose=True):
    """
    Generate embeddings for all sequences in a FASTA file.
    
    Args:
        input_fasta: Path to input FASTA file
        model: ProtAlbert model
        tokenizer: ProtAlbert tokenizer
        device: 'cuda' or 'cpu'
        verbose: Whether to print progress
        
    Returns:
        tuple: (embeddings_dict, sequences_dict)
    """
    embeddings = {}
    sequences = {}
    
    # Count total sequences
    total = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    start_time = time.time()

    for count, record in enumerate(SeqIO.parse(input_fasta, "fasta"), 1):
        sequences[record.id] = record
        emb = generate_embedding(str(record.seq), model, tokenizer, device)
        embeddings[record.id] = emb
        
        if verbose:
            # Estimate remaining time
            elapsed_time = time.time() - start_time
            progress = count / total
            remaining_time = (elapsed_time / progress) - elapsed_time if progress > 0 else 0
            
            print(f"\rProgress: {progress*100:.2f}% | Elapsed: {elapsed_time:.2f}s | "
                  f"Estimated Remaining: {remaining_time:.2f}s", end="")
    
    if verbose:
        print()  # Newline after progress
    
    return embeddings, sequences
