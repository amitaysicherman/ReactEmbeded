import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from contrastive_learning.dataset import ReactionGraphDataset
from contrastive_learning.model import ReactEmbedConfig, EnhancementModule

# --- Model Dims (from old code) ---
model_to_dim = {
    "ProtBert": 1024,
    "MolFormer": 768,
    # Add other models as needed
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def model_args_to_name(p_model, m_model, n_layers, hidden_dim, shared_dim, lr, margin, alpha):
    """Creates a descriptive name for saving the model."""
    return f"{p_model}-{m_model}-{n_layers}-{hidden_dim}-{shared_dim}-{lr}-{margin}-{alpha}"

def run_epoch(model, loader, optimizer, loss_fn, alpha, is_train):
    if is_train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0
    total_intra_loss = 0
    total_cross_loss = 0
    count = 0

    for embeddings, types in tqdm(loader, desc="Epoch"):
        # Move data to device
        h_a = embeddings["anchor"].to(device)
        h_p = embeddings["pos"].to(device)
        h_ni = embeddings["neg_intra"].to(device)
        h_nc = embeddings["neg_cross"].to(device)
        
        # Get entity types (all are batches, so just check first element)
        t_a, t_p = types["anchor"][0], types["pos"][0]
        t_ni, t_nc = types["neg_intra"][0], types["neg_cross"][0]
        
        # Project embeddings into shared space
        h_a_shared = model(h_a, t_a)
        h_p_shared = model(h_p, t_p)
        h_ni_shared = model(h_ni, t_ni)
        h_nc_shared = model(h_nc, t_nc)

        # Calculate dual losses (as per paper Eq 3, 4, 5)
        loss_intra = loss_fn(h_a_shared, h_p_shared, h_ni_shared)
        loss_cross = loss_fn(h_a_shared, h_p_shared, h_nc_shared)
        
        total_loss_batch = alpha * loss_intra + (1 - alpha) * loss_cross
        
        if is_train:
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
        total_loss += total_loss_batch.item()
        total_intra_loss += loss_intra.item()
        total_cross_loss += loss_cross.item()
        count += 1
        
    avg_loss = total_loss / count
    avg_intra = total_intra_loss / count
    avg_cross = total_cross_loss / count
    
    return avg_loss, avg_intra, avg_cross


def main(data_name, p_model, m_model, shared_dim, n_layers, hidden_dim,
         dropout, epochs, lr, margin, alpha, batch_size):

    # --- Setup Model Saving ---
    name = model_args_to_name(p_model, m_model, n_layers, hidden_dim, shared_dim, lr, margin, alpha)
    save_dir = f"data/{data_name}/model/{name}/"
    model_file = f"{save_dir}/model.pt"
    config_file = f"{save_dir}/config.txt"

    if os.path.exists(model_file):
        print(f"Model already exists at {model_file}. Skipping training.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # --- Setup Dataset & DataLoader ---
    # Note: Using num_samples for an "epoch" as defined in the dataset
    train_dataset = ReactionGraphDataset(data_name, p_model, m_model, k_hops=[2, 3, 4, 5])
    # Using a simple shuffle in the loader. The dataset __getitem__ handles the complex sampling.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --- Setup Model ---
    p_dim = model_to_dim[p_model]
    m_dim = model_to_dim[m_model]
    
    config = ReactEmbedConfig(p_dim, m_dim, shared_dim, n_layers, hidden_dim, dropout)
    model = EnhancementModule(config).to(device)
    config.save_to_file(config_file)
    
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Setup Loss & Optimizer ---
    # Use Cosine Distance as per paper (1 - cosine_similarity)
    cosine_distance = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    contrastive_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance, 
        margin=margin
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Training Loop ---
    best_loss = float("inf")
    print("--- Starting Training ---")
    for epoch in range(epochs):
        train_loss, intra_loss, cross_loss = run_epoch(
            model, train_loader, optimizer, contrastive_loss, alpha, is_train=True
        )
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {train_loss:.4f} "
              f"| Intra Loss: {intra_loss:.4f} | Cross Loss: {cross_loss:.4f}")

        # Save the best model based on training loss
        # The paper uses a validation set of reactions, but for simplicity,
        # we save the best model on the training objective.
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), model_file)
            print(f"New best model saved to {model_file}")
            
    print("--- Training Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ReactEmbed Contrastive Learning')
    # Data & Model Args
    parser.add_argument('--data_name', type=str, default="reactome")
    parser.add_argument('--p_model', type=str, default="ProtBert", choices=["ProtBert"])
    parser.add_argument('--m_model', type=str, default="MolFormer", choices=["MolFormer"])
    # Model Architecture Args (from paper)
    parser.add_argument('--shared_dim', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1) # Added sensible default
    # Training Hyperparameters (from paper)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--margin', type=float, default=0.1, help="Triplet loss margin")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for L_intra (vs L_cross)")
    
    args = parser.parse_args()

    main(data_name=args.data_name, p_model=args.p_model, m_model=args.m_model,
         shared_dim=args.shared_dim, n_layers=args.n_layers, hidden_dim=args.hidden_dim,
         dropout=args.dropout, epochs=args.epochs, lr=args.lr, margin=args.margin,
         alpha=args.alpha, batch_size=args.batch_size)