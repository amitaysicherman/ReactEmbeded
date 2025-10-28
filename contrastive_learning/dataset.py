# python
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
    - Filters out nodes with all-zero embeddings.
    - Implements __getitem__ to perform:
        1. Hub-Dampened Positive Sampling (PPMI-weighted 1-hop)
        2. Graph-Based Hard Negative Sampling (k-hop, intra- and cross-domain)
    """

    def __init__(self, data_name, p_model, m_model, k_hops=[2, 3, 4, 5], num_samples=1_000):
        self.data_path = f"data/{data_name}"
        self.k_hops = k_hops
        self.num_samples = num_samples  # Number of samples per epoch

        print("Loading PPMI graph...")
        graph_file = os.path.join(self.data_path, "reaction_graph.gpickle")
        with open(graph_file, "rb") as f:
            self.graph = pickle.load(f)

        print("Loading pre-trained embeddings...")
        self.p_embeds = np.load(os.path.join(self.data_path, f"{p_model}_vectors.npy"))
        self.m_embeds = np.load(os.path.join(self.data_path, f"{m_model}_vectors.npy"))

        # Start with all graph nodes, then filter out nodes with zero embeddings
        all_nodes = list(self.graph.nodes)
        self.nodes = []
        # Temporarily populate node_to_idx for _is_zero_embedding to function
        # (we'll overwrite it after filtering)
        for node in all_nodes:
            if not self._is_zero_embedding_static(node):
                self.nodes.append(node)

        removed = len(all_nodes) - len(self.nodes)
        print(f"Filtered out {removed} nodes with all-zero embeddings; {len(self.nodes)} nodes remain.")

        if not self.nodes:
            raise RuntimeError("No nodes with non-zero embeddings available after filtering.")

        self.node_set = set(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        # Pre-calculate k-hop neighborhoods for efficiency (only include valid nodes)
        print(f"Pre-calculating k-hop neighborhoods (max k={max(k_hops)})...")
        self.k_hop_neighbors = {}
        for node in tqdm(self.nodes):
            self.k_hop_neighbors[node] = self._get_k_hop_neighbors(node, max(k_hops))

        # For fallback sampling (only include nodes with valid embeddings)
        self.protein_nodes = [n for n in self.nodes if self.graph.nodes[n].get('type') == 'protein' or self.graph.nodes[n].get('type') == 'P']
        self.molecule_nodes = [n for n in self.nodes if self.graph.nodes[n].get('type') == 'molecule' or self.graph.nodes[n].get('type') == 'M']

    def _is_zero_embedding_static(self, node_id):
        """
        Static helper used during init filtering (calls underlying arrays directly).
        Returns True if the embedding is all zeros.
        """
        try:
            node_type, idx = node_id.split('_')
            idx = int(idx)
        except Exception:
            return True  # treat malformed ids as invalid

        if node_type == 'P':
            if idx < 0 or idx >= len(self.p_embeds):
                return True
            return np.allclose(self.p_embeds[idx], 0.0)
        elif node_type == 'M':
            if idx < 0 or idx >= len(self.m_embeds):
                return True
            return np.allclose(self.m_embeds[idx], 0.0)
        else:
            return True

    def _get_k_hop_neighbors(self, start_node, max_k):
        """
        Gets all nodes at distances 1 to max_k but only returns nodes
        that passed the embedding filter (are present in self.node_set).
        Returns a dictionary: {1: {nodes at 1-hop}, 2: {nodes at 2-hop}, ...}
        """
        hops = {}
        visited = {start_node}
        current_level = {start_node}

        for k in range(1, max_k + 1):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in self.node_set and neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            hops[k] = next_level
            current_level = next_level
            if not current_level:  # Stop if no new nodes
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
        if fallback_list:
            return random.choice(fallback_list)
        # As an ultimate fallback, pick any node from available nodes
        return random.choice(self.nodes)

    def __getitem__(self, idx):
        # 1. Sample Anchor
        anchor_node = random.choice(self.nodes)
        anchor_emb, anchor_type = self._get_node_embedding(anchor_node)

        # 2. Hub-Dampened Positive Sampling (PPMI-weighted 1-hop)
        neighbors = [n for n in self.graph.neighbors(anchor_node) if n in self.node_set]
        if not neighbors:
            # Handle disconnected nodes: sample a random positive of the other type
            if anchor_type == 'P':
                pos_node = random.choice(self.molecule_nodes) if self.molecule_nodes else random.choice(self.nodes)
            else:
                pos_node = random.choice(self.protein_nodes) if self.protein_nodes else random.choice(self.nodes)
        else:
            weights = [self.graph[anchor_node][neighbor].get('weight', 1.0) for neighbor in neighbors]
            pos_node = random.choices(neighbors, weights=weights, k=1)[0]

        pos_emb, pos_type = self._get_node_embedding(pos_node)

        # 3. Graph-Based Hard Negative Sampling (k-hop)
        k = random.choice(self.k_hops)

        # Get all nodes at exactly k-hops (may be empty)
        k_hop_nodes = set()
        if anchor_node in self.k_hop_neighbors:
            k_hop_nodes = self.k_hop_neighbors[anchor_node].get(k, set())

        # Ensure we don't accidentally sample the positive or anchor
        k_hop_nodes = k_hop_nodes - set(neighbors) - {anchor_node, pos_node}

        # Split k-hop nodes by domain
        neg_intra_candidates = []
        neg_cross_candidates = []
        for node in k_hop_nodes:
            ntype = self.graph.nodes[node].get('type')
            # normalize possible type spellings
            if ntype in ('P', 'protein'):
                ntype_norm = 'P'
            elif ntype in ('M', 'molecule'):
                ntype_norm = 'M'
            else:
                # If type is missing, try to infer from node prefix
                ntype_norm = 'P' if node.startswith('P_') else 'M'
            if ntype_norm == anchor_type:
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
        type_map = {"P": 0, "M": 1}
        types = {
            "anchor": torch.tensor(type_map[anchor_type], dtype=torch.long),
            "pos": torch.tensor(type_map[pos_type], dtype=torch.long),
            "neg_intra": torch.tensor(type_map[neg_intra_type], dtype=torch.long),
            "neg_cross": torch.tensor(type_map[neg_cross_type], dtype=torch.long)
        }
        embeddings = {
            "anchor": torch.tensor(anchor_emb, dtype=torch.float),
            "pos": torch.tensor(pos_emb, dtype=torch.float),
            "neg_intra": torch.tensor(neg_intra_emb, dtype=torch.float),
            "neg_cross": torch.tensor(neg_cross_emb, dtype=torch.float)
        }

        return embeddings, types