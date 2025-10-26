import os
import time
import xml.etree.ElementTree as ET
import pybiopax
import requests
from tqdm import tqdm
import networkx as nx
import numpy as np
from collections import Counter
import argparse
import pickle

# --- Entity Parsing and Sequence Fetching (from old code) ---

PROTEIN = "protein"
MOLECULE = "molecule"

def db_to_type(db_name):
    proteins_data_bases = ["uniprot"]
    molecules_data_bases = ["chebi", "pubchem compound", "guide to pharmacology"]
    db_name = db_name.lower()
    if db_name in proteins_data_bases:
        return PROTEIN
    elif db_name in molecules_data_bases:
        return MOLECULE

def get_req(url: str, to_json=False, ret=3):
    for i in range(ret):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json() if to_json else response.text
            print(f"Failed to retrieve url ({i}): {url}, Status: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed ({i}): {e}")
        time.sleep(2)
    return {} if to_json else ""

def from_second_line(seq):
    lines = seq.split("\n")
    return "".join(lines[1:]) if len(lines) > 1 else ""

def get_smiles_from_chebi(chebi_id, default_seq=""):
    chebi_id = chebi_id.replace("CHEBI:", "")
    url = f"https://www.ebi.ac.uk/chebi/saveStructure.do?xml=true&chebiId={chebi_id}&imageId=0"
    response = get_req(url)
    if response:
        try:
            root = ET.fromstring(response)
            smiles_tag = root.find('.//SMILES')
            return smiles_tag.text if smiles_tag is not None else default_seq
        except ET.ParseError:
            print(f"Failed to parse SMILES for {chebi_id}")
    return default_seq

def get_sequence(identifier, db_name):
    db_name = db_name.lower()
    default_seq = ""
    db_handlers = {
        "uniprot": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "chebi": lambda id: get_smiles_from_chebi(id, default_seq),
        "guide to pharmacology": lambda id: get_req(
            f"https://www.guidetopharmacology.org/services/ligands/{id}/structure", to_json=True).get("smiles", default_seq),
        "pubchem compound": lambda id: get_req(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/CanonicalSMILES/TXT",
            to_json=False).strip(),
    }
    handler = db_handlers.get(db_name)
    if handler:
        return handler(identifier)
    return default_seq

def element_parser(element: pybiopax.biopax.PhysicalEntity):
    if not hasattr(element, "entity_reference") or not hasattr(element.entity_reference, "xref"):
        if hasattr(element, "xref"):
            for xref in element.xref:
                if xref.db.lower() == "uniprot" or xref.db.lower() == "chebi":
                    ref_db, ref_id = xref.db, xref.id
                    break
            else:
                ref_db, ref_id = "0", element.display_name
        else:
            ref_db, ref_id = "0", element.display_name
    elif len(element.entity_reference.xref) >= 1:
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id
    else:
        ref_db, ref_id = "0", element.display_name
    return ref_db, ref_id

def get_all_elements(entity):
    elements = []
    if entity.member_physical_entity:
        for entity_member in entity.member_physical_entity:
            elements.extend(get_all_elements(entity_member))
    if isinstance(entity, pybiopax.biopax.Complex):
        for component in entity.component:
            elements.extend(get_all_elements(component))
    elif isinstance(entity, pybiopax.biopax.PhysicalEntity):
        elements.append(element_parser(entity))
    return elements

def elements_to_ids(elements, proteins_to_id, molecules_to_id, entity_type_map):
    protein_ids = []
    molecule_ids = []
    for db, db_id in elements:
        type_ = db_to_type(db)
        key = (db, db_id)
        if type_ == PROTEIN:
            if key not in proteins_to_id:
                proteins_to_id[key] = len(proteins_to_id)
            node_id = f"P_{proteins_to_id[key]}"
            protein_ids.append(node_id)
            entity_type_map[node_id] = PROTEIN
        elif type_ == MOLECULE:
            if key not in molecules_to_id:
                molecules_to_id[key] = len(molecules_to_id)
            node_id = f"M_{molecules_to_id[key]}"
            molecule_ids.append(node_id)
            entity_type_map[node_id] = MOLECULE
    return protein_ids, molecule_ids

def save_sequences(data_dict, output_file):
    from joblib import Parallel, delayed
    all_seq = Parallel(n_jobs=-1)(
        delayed(get_sequence)(db_id, db) for (db, db_id), _ in tqdm(data_dict.items(), desc=f"Fetching {output_file}"))
    with open(output_file, "w") as f:
        f.write("\n".join(all_seq))

# --- NEW PPMI Graph Construction (from paper) ---

def build_ppmi_graph(co_occurrence_counts, entity_type_map):
    """Builds a NetworkX graph with PPMI edge weights."""
    G = nx.Graph()
    total_co_occurrences = sum(co_occurrence_counts.values())
    
    # Calculate marginal occurrences D(e_i)
    marginal_counts = Counter()
    for (e_i, e_j), count in co_occurrence_counts.items():
        marginal_counts[e_i] += count
        marginal_counts[e_j] += count
        
    # Add nodes with their type
    for node, type_ in entity_type_map.items():
        G.add_node(node, type=type_)

    print("Calculating PPMI weights...")
    for (e_i, e_j), C_ei_ej in tqdm(co_occurrence_counts.items()):
        if C_ei_ej == 0:
            continue
            
        D_ei = marginal_counts[e_i]
        D_ej = marginal_counts[e_j]

        P_ei_ej = C_ei_ej / total_co_occurrences
        P_ei = D_ei / total_co_occurrences
        P_ej = D_ej / total_co_occurrences

        if P_ei * P_ej == 0:
            continue
            
        pmi = np.log(P_ei_ej / (P_ei * P_ej))
        ppmi = max(0, pmi)

        if ppmi > 0:
            G.add_edge(e_i, e_j, weight=ppmi)
            
    return G

# --- Main Execution ---

def main(data_name, input_owl_file):
    proteins_to_id = {}
    molecules_to_id = {}
    entity_type_map = {} # Map node ID (e.g., "P_10") to type ("protein")
    
    output_base = f"data/{data_name}"
    os.makedirs(output_base, exist_ok=True)
    
    proteins_file = f"{output_base}/proteins.txt"
    molecules_file = f"{output_base}/molecules.txt"
    graph_file = f"{output_base}/reaction_graph.gpickle"
    
    # Check if graph already exists
    if os.path.exists(graph_file):
        print(f"Graph file {graph_file} already exists. Skipping preprocessing.")
        return

    print(f"Loading BioPAX model from {input_owl_file}...")
    model = pybiopax.model_from_owl_file(input_owl_file)
    all_reactions = model.get_objects_by_type(pybiopax.biopax.BiochemicalReaction)
    
    co_occurrence_counts = Counter()
    
    print(f"Parsing {len(all_reactions)} reactions...")
    for reaction in tqdm(all_reactions):
        elements = []
        for entity in reaction.left:
            elements.extend(get_all_elements(entity))
        for entity in reaction.right:
            elements.extend(get_all_elements(entity))
            
        protein_ids, molecule_ids = elements_to_ids(elements, proteins_to_id, molecules_to_id, entity_type_map)
        
        # Get unique participants for this reaction
        all_participants = list(set(protein_ids + molecule_ids))
        
        # Update co-occurrence counts
        for i in range(len(all_participants)):
            for j in range(i + 1, len(all_participants)):
                # Ensure canonical order (e.g., P_1, M_10)
                pair = tuple(sorted((all_participants[i], all_participants[j])))
                co_occurrence_counts[pair] += 1

    # --- Build and Save PPMI Graph ---
    print("Building PPMI-weighted graph...")
    G = build_ppmi_graph(co_occurrence_counts, entity_type_map)
    with open(graph_file, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved PPMI graph to {graph_file}")
    print(f"Graph stats: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # --- Save Sequence Files ---
    with open(proteins_file, "w") as f:
        for (db, db_id), idx in proteins_to_id.items():
            f.write(f'P_{idx},{db},{db_id}\n')
            
    with open(molecules_file, "w") as f:
        for (db, db_id), idx in molecules_to_id.items():
            f.write(f'M_{idx},{db},{db_id}\n')
            
    save_sequences(proteins_to_id, proteins_file.replace(".txt", "_sequences.txt"))
    save_sequences(molecules_to_id, molecules_file.replace(".txt", "_sequences.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse BioPAX and build PPMI graph.')
    parser.add_argument('--data_name', type=str, default="reactome", help='Dataset name (e.g., reactome)')
    parser.add_argument('--input_owl_file', type=str, required=True, help='Path to the .owl BioPAX file')
    args = parser.parse_args()
    
    main(args.data_name, args.input_owl_file)