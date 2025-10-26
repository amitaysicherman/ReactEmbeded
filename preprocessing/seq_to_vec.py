import io
import os
import re
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoModel, BertModel, BertTokenizer, AutoTokenizer
import argparse

# --- Setup Device ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Model Definitions ---

class PortBert:
    """ProtBert model from Rostlab"""
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert").to(device).eval()
        self.dim = 1024

    def to_vec(self, seq: str):
        if len(seq) > 1023:
            seq = seq[:1023]
        # Replace rare amino acids
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
        ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Get mean embedding of last hidden state
        vec = embedding_repr.last_hidden_state[0].mean(dim=0)
        return vec.detach().cpu().numpy().flatten()


class MoLFormer:
    """MoLFormer model from IBM"""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                               trust_remote_code=True).to(device).eval()
        self.dim = 768

    def to_vec(self, seq: str):
        if len(seq) > 510: # Max sequence length for MolFormer
            seq = seq[:510]
        inputs = self.tokenizer([seq], return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the pooler output as the embedding
        vec = outputs.pooler_output
        return vec.detach().cpu().numpy().flatten()

# --- Main SeqToVec Class ---

class SeqToVec:
    """Wrapper class to select and use an embedding model"""
    def __init__(self, model_name):
        self.mem = dict() # Simple cache
        if model_name == "ProtBert":
            self.model = PortBert()
            self.dtype = "protein"
            self.dim = self.model.dim
        elif model_name == "MolFormer":
            self.model = MoLFormer()
            self.dtype = "molecule"
            self.dim = self.model.dim
        else:
            raise ValueError(f"Unknown model: {model_name}. This simplified script only supports 'ProtBert' and 'MolFormer'.")

    def to_vec(self, seq: str):
        if len(seq) == 0:
            return None
        if self.dtype == "protein":
            seq = seq.replace(".", "") # Clean protein sequences
        
        if seq in self.mem:
            return self.mem[seq]
        
        try:
            vec = self.model.to_vec(seq)
            self.mem[seq] = vec
            return vec
        except Exception as e:
            print(f"Error processing sequence '{seq[:20]}...': {e}")
            return None

    def lines_to_vecs(self, lines):
        all_vecs = []
        for line in tqdm(lines, desc=f"Generating embeddings"):
            seq = line.strip()
            if len(seq) == 0:
                all_vecs.append(None)
                continue
            vec = self.to_vec(seq)
            all_vecs.append(vec)
            
        # Fill None with zeros
        zero_vec = np.zeros(self.dim)
        for i, vec in enumerate(all_vecs):
            if vec is None:
                all_vecs[i] = zero_vec
                
        return np.array(all_vecs)

# --- Helper Functions ---

def model_to_type(model_name):
    if model_name in ["MolFormer"]:
        return "molecule"
    elif model_name in ["ProtBert"]:
        return "protein"
    else:
        raise ValueError(f"Unknown model: {model_name}")

# --- Main Execution ---

def main(model, data_name):
    base_path = f'data/{data_name}/'
    data_type = model_to_type(model)
    
    if data_type == "protein":
        file_path = os.path.join(base_path, "proteins_sequences.txt")
    else:
        file_path = os.path.join(base_path, "molecules_sequences.txt")
        
    output_file = os.path.join(base_path, f"{model}_vectors.npy")
    
    if os.path.exists(output_file):
        print(f"Embeddings file already exists: {output_file}")
        return

    if not os.path.exists(file_path):
        print(f"Error: Sequence file not found at {file_path}")
        print("Please run preprocessing/biopax_parser.py first.")
        return

    print(f"Loading sequences from {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    print(f"Initializing {model} model...")
    seq_to_vec = SeqToVec(model)
    
    all_vecs = seq_to_vec.lines_to_vecs(lines)
    
    np.save(output_file, all_vecs)
    print(f"Saved {all_vecs.shape} embeddings to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert sequences to vector embeddings.')
    parser.add_argument('--model', type=str, required=True, choices=["ProtBert", "MolFormer"],
                        help='Model to use for embedding.')
    parser.add_argument('--data_name', type=str, default="reactome",
                        help='Data name directory (e.g., reactome)')
    args = parser.parse_args()
    
    main(args.model, args.data_name)