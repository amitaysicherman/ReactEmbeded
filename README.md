# ReactEmbed: A Plug-and-Play Module for Unifying Protein-Molecule Representations Guided by Biochemical Reaction Networks

This repository contains the official implementation of the ReactEmbed framework.

ReactEmbed is a lightweight, plug-and-play enhancement module that aligns existing, frozen embeddings from state-of-the-art protein and molecule models into a unified, functionally-aware space. It leverages biochemical reaction networks as a definitive source of functional semantics, based on the principle that co-participation in a reaction defines a shared functional role.

## Framework Overview

The ReactEmbed framework operates in three main stages:

1.  **Graph Construction:** We process a database of biochemical reactions (e.g., Reactome) into a weighted, undirected graph. To capture functional specificity and dampen the effect of ubiquitous "hub" nodes (like ATP), we use **Positive Pointwise Mutual Information (PPMI)** as the edge weight between any two co-participating entities.
2.  **Relational Learning:** We train a lightweight enhancement module, consisting of two MLPs (P2U for proteins, M2U for molecules), to project frozen, pre-trained embeddings (e.g., from ProtBert and MolFormer) into a shared space. The training uses a novel relational learning strategy:
    * **Hub-Dampened Positive Sampling:** An anchor node samples a positive partner from its 1-hop neighborhood with a probability proportional to their PPMI edge weight.
    * **Graph-Based Hard Negative Sampling:** The anchor samples two "hard" negatives from its $k$-hop neighborhood ($k > 1$): one from the same domain (e.g., another protein) and one from the opposite domain (e.g., a molecule).
    * **Dual-Loss Objective:** We optimize two triplet losses: $\mathcal{L}_{\text{intra}}$ (preserving same-domain structure) and $\mathcal{L}_{\text{cross}}$ (driving cross-domain alignment), combined as $\mathcal{L}_{\text{total}} = \alpha\mathcal{L}_{\text{intra}} + (1-\alpha)\mathcal{L}_{\text{cross}}$.
3.  **Downstream Task Evaluation:** The resulting frozen ReactEmbed-enhanced embeddings are used to train a simple linear classifier (a "linear probe") on downstream tasks, demonstrating improved performance. This unified space also unlocks powerful **zero-shot transfer learning** capabilities.


## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/ReactEmbed.git](https://github.com/your-username/ReactEmbed.git)
    cd ReactEmbed
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage: End-to-End Example

The `run_experiment.sh` script provides a complete pipeline to reproduce the paper's workflow. This example preprocesses the Reactome dataset, trains a ReactEmbed module on ProtBert and MolFormer embeddings, and evaluates it on the BBBP downstream task.

You can run the full pipeline with a single command:
```bash
bash run_experiment.sh
```

### Manual Pipeline Steps

The full pipeline consists of 4 steps, which are automated in the script above.

**Step 1: Preprocess Reaction Data**
This script parses the raw reaction data (e.g., from Reactome), builds the co-occurrence graph, calculates PPMI weights, and saves the final graph. It also extracts all protein and molecule sequences.

```bash
python preprocessing/biopax_parser.py \
    --data_name reactome \
    --input_owl_file /path/to/Homo_sapiens.owl
```

**Step 2: Generate Pre-trained Embeddings**
This script uses the sequence files from Step 1 to generate the frozen, pre-trained embeddings that ReactEmbed will align.

```bash
# Generate protein embeddings
python preprocessing/seq_to_vec.py --model ProtBert --data_name reactome

# Generate molecule embeddings
python preprocessing/seq_to_vec.py --model MolFormer --data_name reactome
```

**Step 3: Train ReactEmbed Module**
This is the core of the framework. It loads the PPMI graph and the frozen embeddings and trains the enhancement module using our advanced relational learning strategy.

```bash
python contrastive_learning/trainer.py \
    --data_name reactome \
    --p_model ProtBert \
    --m_model MolFormer \
    --shared_dim 768 \
    --hidden_dim 1024 \
    --n_layers 2 \
    --epochs 50 \
    --batch_size 256 \
    --lr 5e-5 \
    --margin 0.1 \
    --alpha 0.5
```

**Step 4: Evaluate on a Downstream Task**
First, prepare the specific task data (e.g., BBBP). This script downloads the data, splits it, and generates the necessary embedding files for the task.

```bash
python eval_tasks/prep_task.py \
    --task_name BBBP \
    --p_model ProtBert \
    --m_model MolFormer
```

Finally, train and evaluate a linear probe on the downstream task using the ReactEmbed-enhanced embeddings.

```bash
# Get the path to the trained ReactEmbed model from Step 3
MODEL_NAME="ProtBert-MolFormer-..." # Construct from params
FUSION_PATH="data/reactome/model/${MODEL_NAME}/"

python eval_tasks/trainer.py \
    --task_name BBBP \
    --p_model ProtBert \
    --m_model MolFormer \
    --fusion_name ${FUSION_PATH} \
    --use_fuse 1 \
    --use_model 0 \
    --metric auc
```