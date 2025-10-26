import os
import numpy as np
from torchdrug import datasets
from torchdrug.data import ordered_scaffold_split
from torchdrug.transforms import ProteinView
from tqdm import tqdm
import argparse

# Import the SeqToVec class from the preprocessing directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.seq_to_vec import SeqToVec, model_to_type

from eval_tasks.models import DataType # Using the DataType enum from your old code
from eval_tasks.tasks import name_to_task # Using the Task definitions from your old code

base_dir = "data/torchdrug/"
os.makedirs(base_dir, exist_ok=True)

# --- Helper Functions (from old code) ---

def task_name_to_dataset_class(task_name):
    if task_name.startswith("GeneOntology"):
        return getattr(datasets, "GeneOntology")
    return getattr(datasets, task_name)

def get_seq(x):
    try:
        if hasattr(x, "to_sequence"): # Protein
            return x.to_sequence().replace(".G", "")
        else: # Molecule
            return x.to_smiles()
    except Exception as e:
        print(e)
        return None

# --- Main Prep Script ---

def prep_dataset_seqs(task):
    """
    Downloads and splits the torchdrug dataset, saving sequences to text files.
    This is adapted from your `prep_tasks_seqs.py`.
    """
    output_base = os.path.join(base_dir, task.name)
    os.makedirs(output_base, exist_ok=True)
    labels_file = os.path.join(output_base, "train_labels.txt")
    
    if os.path.exists(labels_file):
        print(f"Task sequences for {task.name} already exist. Skipping download.")
        return

    print(f"Preparing sequences for task: {task.name}")
    
    # Setup dataset args
    if task.dtype1 == DataType.PROTEIN:
        keys = ["graph"] if task.dtype2 is None else ["graph1"]
        args = dict(transform=ProteinView(view="residue", keys=keys),
                    atom_feature=None, bond_feature=None)
    else:
        args = dict()
    if task.branch is not None:
        args["branch"] = task.branch

    dataset_class = task_name_to_dataset_class(task.name)
    dataset = dataset_class(os.path.join(base_dir, task.name), **args)
    
    labels_keys = getattr(dataset_class, 'target_fields', ["targets"])

    try:
        splits = dataset.split()
        train, valid, test = splits[:3]
    except Exception:
        print("Using ordered scaffold split")
        train, valid, test = ordered_scaffold_split(dataset, None)

    for split, name in zip([train, valid, test], ["train", "valid", "test"]):
        all_seq_1 = []
        all_labels = []
        
        for i in tqdm(range(len(split)), desc=f"Processing {name} split"):
            key1 = "graph" if task.dtype2 is None else "graph1"
            seq_1 = get_seq(split[i][key1])
            if seq_1 is None:
                continue
            all_seq_1.append(seq_1)

            if labels_keys == "targets": # Handle GO/EC tasks
                new_labels = " ".join(str(x) for x in split[i][labels_keys].tolist())
            else:
                new_labels = " ".join([str(split[i][key]) for key in labels_keys])
            all_labels.append(new_labels)

        with open(os.path.join(output_base, f"{name}_1.txt"), "w") as f:
            f.write("\n".join(all_seq_1))
        with open(os.path.join(output_base, f"{name}_labels.txt"), "w") as f:
            f.write("\n".join(all_labels))
            
    print(f"Finished preparing sequences for {task.name}")

def prep_dataset_vecs(task, p_model, m_model):
    """
    Converts the sequence .txt files into embedding .npy files.
    This is adapted from your `prep_tasks_vecs.py`.
    """
    task_dir = os.path.join(base_dir, task.name)
    
    # Determine which model to use for which input
    if task.dtype1 == DataType.PROTEIN:
        emb_name_1, model_name_1 = p_model, p_model
    else:
        emb_name_1, model_name_1 = m_model, m_model
        
    # Initialize the correct embedder
    seq2vec_1 = SeqToVec(model_name_1)

    for split in ['train', 'valid', 'test']:
        print(f"Generating vectors for {task.name} - {split} split...")
        
        # --- Process Input 1 ---
        in_file_1 = f"{task_dir}/{split}_1.txt"
        out_file_1 = f"{task_dir}/{split}_{emb_name_1}_1.npy"
        
        if not os.path.exists(out_file_1):
            with open(in_file_1, 'r') as f:
                lines = f.read().splitlines()
            vectors = seq2vec_1.lines_to_vecs(lines)
            np.save(out_file_1, vectors)
            print(f"Saved {split} vectors to {out_file_1}")
        else:
            print(f"{out_file_1} already exists.")

        # --- Process Labels ---
        label_in_file = f"{task_dir}/{split}_labels.txt"
        label_out_file = f"{task_dir}/{split}_labels.npy"
        
        if not os.path.exists(label_out_file):
            with open(label_in_file, 'r') as f:
                lines = f.read().splitlines()
            lines = [line.split() for line in lines]
            labels = np.stack([np.array([float(label) for label in line]) for line in tqdm(lines)])
            labels = np.nan_to_num(labels)
            np.save(label_out_file, labels)
            print(f"Saved {split} labels to {label_out_file}")
        else:
            print(f"{label_out_file} already exists.")
            
    print(f"Finished preparing vectors for {task.name}")


def main(task_name, p_model, m_model):
    task = name_to_task[task_name]
    
    # Step 1: Download data and create sequence files
    prep_dataset_seqs(task)
    
    # Step 2: Convert sequence files to embedding files
    prep_dataset_vecs(task, p_model, m_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare downstream task data.')
    parser.add_argument("--task_name", type=str, default="BBBP")
    parser.add_argument('--p_model', type=str, default="ProtBert")
    parser.add_argument('--m_model', type=str, default="MolFormer")
    args = parser.parse_args()
    
    main(args.task_name, args.p_model, args.m_model)