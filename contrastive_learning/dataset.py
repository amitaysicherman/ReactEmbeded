import os
import pickle
import random
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ReactionGraphDataset(Dataset):
    """
    Implements the advanced sampling strategy from the ReactEmbed paper (Algorithm 1).
    - Loads the PPMI-weighted graph.
    - Loads the frozen pre-trained embeddings.
    - Implements __getitem__ to perform:
        1. Hub-Dampened Positive Sampling (PPMI-weighted 1-hop)
        2. Graph-Based Hard Negative Sampling (k-hop, intra- and cross-domain)
    """
    def __init__(self, data_name, p_model, m_model, k_hops=[2, 3, 4, 5], num_samples=1_000_000):
        self.data_path = f"data/{data_name}"
        self.k_hops = k_hops
        self.num_samples = num_samples # Number of samples per epoch

        print("Loading PPMI graph...")
        graph_file = os.path.join(self.data_path, "reaction_graph.gpickle")
        with open(graph_file, "rb") as f:
            self.graph = pickle.load(f)
            
        print("Loading pre-trained embeddings...")
        self.p_embeds = np.load(os.path.join(self.data_path, f"{p_model}_vectors.npy"))
        self.m_embeds = np.load(os.path.join(self.data_path, f"{m_model}_vectors.npy"))
        
        self.nodes = list(self.graph.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Pre-calculate k-hop neighborhoods for efficiency
        print(f"Pre-calculating k-hop neighborhoods (max k={max(k_hops)})...")
        self.k_hop_neighbors = {}
        for node in tqdm(self.nodes):
            self.k_hop_neighbors[node] = self._get_k_hop_neighbors(node, max(k_hops))
            
        # For fallback sampling
        self.protein_nodes = [n for n in self.nodes if self.graph.nodes[n]['type'] == 'protein']
        self.molecule_nodes = [n for n in self.nodes if self.graph.nodes[n]['type'] == 'molecule']

    def _get_k_hop_neighbors(self, start_node, max_k):
        """
        Gets all nodes at distances 1 to max_k.
        Returns a dictionary: {1: {nodes at 1-hop}, 2: {nodes at 2-hop}, ...}
        """
        hops = {}
        visited = {start_node}
        current_level = {start_node}
        
        for k in range(1, max_k + 1):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            hops[k] = next_level
            current_level = next_level
            if not current_level: # Stop if no new nodes
                break
        return hops
        
    def __len__(self):
        # We define an "epoch" as a fixed number of sampling steps
        return self.num_samples

    def _get_node_embedding(self, node_id):
        """Helper to get the correct embedding vector from the node ID string"""
        node_type, idx = node_id.split('_')
        idx = int(idx)
        if node_type == 'P':
            return self.p_embeds[idx], 'P'
        elif node_type == 'M':
            return self.m_embeds[idx], 'M'
        else:
            raise ValueError(f"Invalid node ID: {node_id}")
            
    def _sample_from_list(self, node_list, fallback_list):
        """Safely samples from a list, using a fallback if the list is empty."""
        if node_list:
            return random.choice(node_list)
        return random.choice(fallback_list)

    def __getitem__(self, idx):
        # 1. Sample Anchor
        anchor_node = random.choice(self.nodes)
        anchor_emb, anchor_type = self._get_node_embedding(anchor_node)
        
        # 2. Hub-Dampened Positive Sampling (PPMI-weighted 1-hop)
        neighbors = list(self.graph.neighbors(anchor_node))
        if not neighbors:
            # Handle disconnected nodes: sample a random positive of the other type
            # This is a fallback, but better than crashing
            if anchor_type == 'P':
                pos_node = random.choice(self.molecule_nodes)
            else:
                pos_node = random.choice(self.protein_nodes)
        else:
            weights = [self.graph[anchor_node][neighbor]['weight'] for neighbor in neighbors]
            pos_node = random.choices(neighbors, weights=weights, k=1)[0]
            
        pos_emb, pos_type = self._get_node_embedding(pos_node)

        # 3. Graph-Based Hard Negative Sampling (k-hop)
        k = random.choice(self.k_hops)
        
        # Get all nodes at exactly k-hops
        k_hop_nodes = self.k_hop_neighbors[anchor_node].get(k, set())
        # Ensure we don't accidentally sample the positive or anchor
        k_hop_nodes = k_hop_nodes - set(neighbors) - {anchor_node}

        # Split k-hop nodes by domain
        neg_intra_candidates = []
        neg_cross_candidates = []
        for node in k_hop_nodes:
            if self.graph.nodes[node]['type'] == anchor_type:
                neg_intra_candidates.append(node)
            else:
                neg_cross_candidates.append(node)
        
        # Sample neg_intra (same domain)
        neg_intra_node = self._sample_from_list(
            neg_intra_candidates,
            self.protein_nodes if anchor_type == 'P' else self.molecule_nodes
        )
        
        # Sample neg_cross (other domain)
        neg_cross_node = self._sample_from_list(
            neg_cross_candidates,
            self.molecule_nodes if anchor_type == 'P' else self.protein_nodes
        )

        neg_intra_emb, neg_intra_type = self._get_node_embedding(neg_intra_node)
        neg_cross_emb, neg_cross_type = self._get_node_embedding(neg_cross_node)
        
        # Types are needed for the model to select the correct MLP (P2U or M2U)
        types = {
            "anchor": anchor_type, "pos": pos_type,
            "neg_intra": neg_intra_type, "neg_cross": neg_cross_type
        }
        
        embeddings = {
            "anchor": torch.tensor(anchor_emb, dtype=torch.float),
            "pos": torch.tensor(pos_emb, dtype=torch.float),
            "neg_intra": torch.tensor(neg_intra_emb, dtype=torch.float),
            "neg_cross": torch.tensor(neg_cross_emb, dtype=torch.float)
        }
        
        return embeddings, types