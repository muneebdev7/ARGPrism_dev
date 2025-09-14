import os
import json
import csv
import torch
import h5py
from Bio import SeqIO
from transformers import AlbertTokenizer, AlbertModel
import subprocess
import time

# Set device to 'cuda' if GPU is available, otherwise use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ProtAlbert model and tokenizer
model_plm = AlbertModel.from_pretrained("Rostlab/prot_albert").to(device)
tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)

# Your trained ARG classifier model
class ARGClassifier(torch.nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 128)
        self.relu2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(128, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.out(x)
        return self.softmax(x)

def generate_embedding(sequence):
    tokens = tokenizer(' '.join(list(sequence)), return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_plm(**tokens)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

def generate_embeddings(input_fasta):
    embeddings = {}
    sequences = {}
    total = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))  # Total number of sequences
    start_time = time.time()  # Record start time

    for count, record in enumerate(SeqIO.parse(input_fasta, "fasta"), 1):
        seq = str(record.seq).replace('U','X').replace('Z','X').replace('O','X')
        sequences[record.id] = record
        emb = generate_embedding(seq)
        embeddings[record.id] = emb
        
        # Estimate the remaining time
        elapsed_time = time.time() - start_time
        progress = count / total
        remaining_time = (elapsed_time / progress) - elapsed_time  # Estimate remaining time
        
        print(f"\rProgress: {progress*100:.2f}% | Elapsed: {elapsed_time:.2f}s | Estimated Remaining: {remaining_time:.2f}s", end="")
    
    print()  # For newline after progress
    return embeddings, sequences

def load_classifier(path):
    clf = ARGClassifier()
    clf.load_state_dict(torch.load(path, map_location=device))
    clf.to(device)
    clf.eval()
    return clf

def classify_embeddings(embeddings, classifier):
    results = {}
    with torch.no_grad():
        for seq_id, emb in embeddings.items():
            emb_tensor = torch.tensor(emb, dtype=torch.float32).to(device)
            output = classifier(emb_tensor.unsqueeze(0))
            pred_class = torch.argmax(output, dim=1).item()
            results[seq_id] = "ARG" if pred_class == 1 else "Non-ARG"
    return results

def save_predicted_ARGs(sequences, predictions, output_fasta):
    arg_records = [sequences[seq_id] for seq_id, pred in predictions.items() if pred=="ARG" and seq_id in sequences]
    SeqIO.write(arg_records, output_fasta, "fasta")
    print(f"Saved {len(arg_records)} predicted ARG sequences to {output_fasta}")

def run_diamond_mapping(predicted_ARGs_fasta, arg_db_fasta, diamond_db_prefix, diamond_output):
    # Build diamond DB if not exists
    if not (os.path.exists(diamond_db_prefix + ".dmnd")):
        cmd_makedb = f"diamond makedb --in {arg_db_fasta} -d {diamond_db_prefix}"
        print(f"Running: {cmd_makedb}")
        subprocess.run(cmd_makedb, shell=True, check=True)
    # Run blastp
    cmd_blastp = f"diamond blastp -q {predicted_ARGs_fasta} -d {diamond_db_prefix} -o {diamond_output} -f 6 qseqid sseqid pident length evalue bitscore"
    print(f"Running: {cmd_blastp}")
    subprocess.run(cmd_blastp, shell=True, check=True)

def load_arg_metadata(metadata_json):
    with open(metadata_json) as f:
        return json.load(f)

def parse_diamond_hits(diamond_tsv):
    best_hits = {}
    with open(diamond_tsv) as f:
        for line in f:
            qid, sid, pid, length, evalue, bitscore = line.strip().split('\t')
            bitscore = float(bitscore)
            if qid not in best_hits or bitscore > best_hits[qid][1]:
                best_hits[qid] = (sid, bitscore)
    return best_hits

def generate_final_report(predictions, best_hits, arg_metadata, output_csv):
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence_ID", "Predicted_Class", "ARG_Ref_ID", "ARG_Name", "Drug"])
        for seq_id, pred_class in predictions.items():
            if pred_class == "ARG":
                if seq_id in best_hits:
                    ref_id = best_hits[seq_id][0]
                    meta = arg_metadata.get(ref_id, {})
                    writer.writerow([seq_id, "ARG", ref_id, meta.get("ARG_name", ""), meta.get("drug", "")])
                else:
                    writer.writerow([seq_id, "ARG", "", "", ""])
            else:
                writer.writerow([seq_id, "Non-ARG", "", "", ""])
    print(f"Final annotated report saved to {output_csv}")

if __name__=="__main__":
    input_fasta = "Input_proteins/ERR589503_PROT.faa"       # Your novel protein FASTA input
    classifier_path = "trained_model/best_model_fold4.pth"
    predicted_ARGs_fasta = "predicted_ARGs.fasta"
    arg_db_fasta = "ARGPrism-DB.fasta"          # ARG reference database FASTA
    diamond_db_prefix = "diamond_arg_db"
    diamond_output = "predicted_ARGs_vs_ref.tsv"
    metadata_json = "metadata_arg.json"  # ARG metadata JSON
    final_report_csv = "final_ARG_prediction_report.csv"

    start_time = time.time()  # Record start time

    print("Generating embeddings for input sequences...")
    embeddings, sequences = generate_embeddings(input_fasta)

    print("Loading classifier...")
    classifier = load_classifier(classifier_path)

    print("Classifying sequences...")
    predictions = classify_embeddings(embeddings, classifier)

    print("Saving predicted ARG sequences...")
    save_predicted_ARGs(sequences, predictions, predicted_ARGs_fasta)

    print("Mapping predicted ARGs to reference ARG database with DIAMOND...")
    run_diamond_mapping(predicted_ARGs_fasta, arg_db_fasta, diamond_db_prefix, diamond_output)

    print("Loading ARG metadata...")
    arg_metadata = load_arg_metadata(metadata_json)

    print("Parsing DIAMOND mapping results...")
    best_hits = parse_diamond_hits(diamond_output)

    print("Generating final annotated report...")
    generate_final_report(predictions, best_hits, arg_metadata, final_report_csv)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nPipeline complete! Total elapsed time: {elapsed_time:.2f} seconds")

