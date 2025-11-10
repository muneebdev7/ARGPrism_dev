"""
Main ARGprism pipeline
"""

import os
import json
import csv
import torch
from Bio import SeqIO
import subprocess

from .classifier import ARGClassifier
from .embeddings import load_protalbert_model, generate_embeddings


class ARGPrismPipeline:
    """
    Complete ARGprism pipeline for ARG prediction and annotation.
    """
    
    def __init__(self, classifier_path, device=None):
        """
        Initialize the pipeline.
        
        Args:
            classifier_path: Path to trained ARG classifier model (.pth file)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading ProtAlbert model...")
        self.model_plm, self.tokenizer = load_protalbert_model(self.device)
        
        print("Loading ARG classifier...")
        self.classifier = self._load_classifier(classifier_path)
    
    def _load_classifier(self, path):
        """Load the trained ARG classifier."""
        clf = ARGClassifier()
        clf.load_state_dict(torch.load(path, map_location=self.device))
        clf.to(self.device)
        clf.eval()
        return clf
    
    def classify_embeddings(self, embeddings):
        """
        Classify protein embeddings as ARG or Non-ARG.
        
        Args:
            embeddings: Dictionary of {seq_id: embedding_array}
            
        Returns:
            Dictionary of {seq_id: "ARG" or "Non-ARG"}
        """
        results = {}
        with torch.no_grad():
            for seq_id, emb in embeddings.items():
                emb_tensor = torch.tensor(emb, dtype=torch.float32).to(self.device)
                output = self.classifier(emb_tensor.unsqueeze(0))
                pred_class = torch.argmax(output, dim=1).item()
                results[seq_id] = "ARG" if pred_class == 1 else "Non-ARG"
        return results
    
    def save_predicted_args(self, sequences, predictions, output_fasta):
        """
        Save predicted ARG sequences to a FASTA file.
        
        Args:
            sequences: Dictionary of {seq_id: SeqRecord}
            predictions: Dictionary of {seq_id: "ARG" or "Non-ARG"}
            output_fasta: Output FASTA file path
        """
        arg_records = [
            sequences[seq_id] 
            for seq_id, pred in predictions.items() 
            if pred == "ARG" and seq_id in sequences
        ]
        SeqIO.write(arg_records, output_fasta, "fasta")
        print(f"Saved {len(arg_records)} predicted ARG sequences to {output_fasta}")
        return len(arg_records)
    
    def run_diamond_mapping(self, query_fasta, db_fasta, db_prefix, output_tsv):
        """
        Map predicted ARGs to reference database using DIAMOND BLAST.
        
        Args:
            query_fasta: Query sequences (predicted ARGs)
            db_fasta: Reference database FASTA
            db_prefix: Database prefix for DIAMOND
            output_tsv: Output TSV file
        """
        # Build DIAMOND database if not exists
        if not os.path.exists(f"{db_prefix}.dmnd"):
            cmd_makedb = f"diamond makedb --in {db_fasta} -d {db_prefix}"
            print(f"Running: {cmd_makedb}")
            subprocess.run(cmd_makedb, shell=True, check=True)
        
        # Run DIAMOND blastp
        cmd_blastp = (
            f"diamond blastp -q {query_fasta} -d {db_prefix} -o {output_tsv} "
            f"-f 6 qseqid sseqid pident length evalue bitscore"
        )
        print(f"Running: {cmd_blastp}")
        subprocess.run(cmd_blastp, shell=True, check=True)
    
    def parse_diamond_hits(self, diamond_tsv):
        """
        Parse DIAMOND output and get best hits.
        
        Args:
            diamond_tsv: DIAMOND output TSV file
            
        Returns:
            Dictionary of {query_id: (subject_id, bitscore)}
        """
        best_hits = {}
        with open(diamond_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                qid, sid, pid, length, evalue, bitscore = parts
                bitscore = float(bitscore)
                
                if qid not in best_hits or bitscore > best_hits[qid][1]:
                    best_hits[qid] = (sid, bitscore)
        return best_hits
    
    def generate_final_report(self, predictions, best_hits, metadata, output_csv):
        """
        Generate final annotated report.
        
        Args:
            predictions: Dictionary of {seq_id: "ARG" or "Non-ARG"}
            best_hits: Dictionary of {seq_id: (ref_id, bitscore)}
            metadata: Dictionary of ARG metadata
            output_csv: Output CSV file path
        """
        with open(output_csv, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Sequence_ID", "Predicted_Class", "ARG_Ref_ID", "ARG_Name", "Drug"])
            
            for seq_id, pred_class in predictions.items():
                if pred_class == "ARG":
                    if seq_id in best_hits:
                        ref_id = best_hits[seq_id][0]
                        meta = metadata.get(ref_id, {})
                        writer.writerow([
                            seq_id, "ARG", ref_id, 
                            meta.get("ARG_name", ""), 
                            meta.get("drug", "")
                        ])
                    else:
                        writer.writerow([seq_id, "ARG", "", "", ""])
                else:
                    writer.writerow([seq_id, "Non-ARG", "", "", ""])
        
        print(f"Final annotated report saved to {output_csv}")
    
    def run(self, input_fasta, arg_db_fasta, metadata_json, output_dir="."):
        """
        Run the complete ARGprism pipeline.
        
        Args:
            input_fasta: Input protein sequences (FASTA)
            arg_db_fasta: ARG reference database (FASTA)
            metadata_json: ARG metadata (JSON)
            output_dir: Output directory
        """
        import time
        start_time = time.time()
        
        # Setup output paths
        predicted_args_fasta = os.path.join(output_dir, "predicted_ARGs.fasta")
        diamond_db_prefix = os.path.join(output_dir, "diamond_arg_db")
        diamond_output = os.path.join(output_dir, "predicted_ARGs_vs_ref.tsv")
        final_report = os.path.join(output_dir, "final_ARG_prediction_report.csv")
        
        # Step 1: Generate embeddings
        print("\n=== Step 1: Generating embeddings ===")
        embeddings, sequences = generate_embeddings(
            input_fasta, self.model_plm, self.tokenizer, self.device
        )
        
        # Step 2: Classify sequences
        print("\n=== Step 2: Classifying sequences ===")
        predictions = self.classify_embeddings(embeddings)
        
        # Step 3: Save predicted ARGs
        print("\n=== Step 3: Saving predicted ARGs ===")
        n_args = self.save_predicted_args(sequences, predictions, predicted_args_fasta)
        
        if n_args > 0:
            # Step 4: Map to reference database
            print("\n=== Step 4: Mapping to reference database ===")
            self.run_diamond_mapping(
                predicted_args_fasta, arg_db_fasta, diamond_db_prefix, diamond_output
            )
            
            # Step 5: Parse DIAMOND results
            print("\n=== Step 5: Parsing DIAMOND results ===")
            best_hits = self.parse_diamond_hits(diamond_output)
        else:
            print("No ARGs predicted. Skipping DIAMOND mapping.")
            best_hits = {}
        
        # Step 6: Load metadata and generate report
        print("\n=== Step 6: Generating final report ===")
        with open(metadata_json) as f:
            metadata = json.load(f)
        
        self.generate_final_report(predictions, best_hits, metadata, final_report)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Pipeline completed successfully!")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"Total sequences processed: {len(predictions)}")
        print(f"Predicted ARGs: {n_args}")
        print(f"Final report: {final_report}")
        print(f"{'='*60}\n")
