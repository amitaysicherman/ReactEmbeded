#!/bin/bash

# This script runs the full ReactEmbed pipeline for a single example.
# It uses ProtBert (protein) and MolFormer (molecule) embeddings,
# trains on the Reactome dataset, and evaluates on the BBBP task.

set -e

# --- CONFIGURATION ---
DATA_NAME="reactome"
OWL_FILE_PATH="data/biopax/Homo_sapiens.owl" # UPDATE THIS PATH
P_MODEL="ProtBert"
M_MODEL="MolFormer"
TASK_NAME="BBBP"
TASK_METRIC="auc"

# CL Model Hyperparameters (from paper)
SHARED_DIM=768
HIDDEN_DIM=1024
N_LAYERS=2
EPOCHS=50
BATCH_SIZE=256
LR=5e-5
MARGIN=0.1
ALPHA=0.5 # Weight for intra/cross loss

# --- STEP 1: PREPROCESS REACTION DATA ---
echo "--- Step 1: Preprocessing Reaction Data (Reactome) ---"
# Download Reactome data if it doesn't exist
mkdir -p data/biopax
if [ ! -f "$OWL_FILE_PATH" ]; then
    echo "Downloading Reactome Homo_sapiens.owl..."
    echo "Please download from https://reactome.org/download-data and place in $OWL_FILE_PATH"
    # Example: wget -O data/biopax/Homo_sapiens.owl https://reactome.org/download/current/biopax.zip
    # You will need to unzip and move the file.
    # For this example, we assume the file exists.
    exit 1
fi

python preprocessing/biopax_parser.py \
    --data_name ${DATA_NAME} \
    --input_owl_file ${OWL_FILE_PATH}


# --- STEP 2: GENERATE PRE-TRAINED EMBEDDINGS ---
echo "--- Step 2: Generating Base Embeddings ---"
echo "Generating ${P_MODEL} embeddings..."
python preprocessing/seq_to_vec.py --model ${P_MODEL} --data_name ${DATA_NAME}

echo "Generating ${M_MODEL} embeddings..."
python preprocessing/seq_to_vec.py --model ${M_MODEL} --data_name ${DATA_NAME}


# --- STEP 3: TRAIN REACTEMBED MODULE ---
echo "--- Step 3: Training ReactEmbed Contrastive Learning Module ---"
python contrastive_learning/trainer.py \
    --data_name ${DATA_NAME} \
    --p_model ${P_MODEL} \
    --m_model ${M_MODEL} \
    --shared_dim ${SHARED_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --n_layers ${N_LAYERS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --margin ${MARGIN} \
    --alpha ${ALPHA}

# Construct model name for path
# Format: {p_model}-{m_model}-{n_layers}-{hidden_dim}-{shared_dim}-{lr}-{margin}-{alpha}
MODEL_NAME="${P_MODEL}-${M_MODEL}-${N_LAYERS}-${HIDDEN_DIM}-${SHARED_DIM}-${LR}-${MARGIN}-${ALPHA}"
FUSION_PATH="data/${DATA_NAME}/model/${MODEL_NAME}/"


# --- STEP 4: EVALUATE ON DOWNSTREAM TASK ---
echo "--- Step 4: Evaluating on Downstream Task (${TASK_NAME}) ---"

echo "Preparing ${TASK_NAME} data..."
python eval_tasks/prep_task.py \
    --task_name ${TASK_NAME} \
    --p_model ${P_MODEL} \
    --m_model ${M_MODEL}

echo "Running ${TASK_NAME} evaluation..."
# Note: use_fuse=1 (use ReactEmbed) and use_model=0 (don't use original)
# To test the baseline, run with use_fuse=0 and use_model=1
python eval_tasks/trainer.py \
    --task_name ${TASK_NAME} \
    --p_model ${P_MODEL} \
    --m_model ${M_MODEL} \
    --fusion_name ${FUSION_PATH} \
    --metric ${TASK_METRIC} \
    --use_fuse 1 \
    --use_model 0 \
    --n_layers 1 \
    --hidden_dim 512 \
    --lr 0.001 \
    --bs 16

echo "--- Experiment Finished ---"