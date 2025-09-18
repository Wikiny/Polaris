import os
import json
import pickle as pkl
import re
import numpy as np
from tqdm import tqdm

import networkx as nx
import dgl
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from collections import defaultdict

# ===================================================================
# 1. Helper functions for semantic embeddings
# ===================================================================

def tokenize(text: str):
    """
    Tokenizer function that splits text by multiple delimiters.
    Delimiters: backslash(\), space( ), dot(.), colon(:), forward slash(/)
    """
    if not isinstance(text, str):
        return []
    
    # Use re.split to split the string and filter out empty strings
    tokens = re.split(r'[/ .:]', text)
    
    return [token.lower() for token in tokens if token]

def get_positional_encoding(max_len, d_model):
    """Generate a positional encoding matrix."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def generate_semantic_embeddings_for_graph(nx_g, w2v_model):
    """
    Computes semantic embeddings for nodes in a graph and adds them
    back as node attributes.
    """
    vector_size = w2v_model.vector_size

    for node_id, data in tqdm(sorted(nx_g.nodes(data=True))):
        semantic_features = data.get('semantic_features', [])
        
        # Default to a zero vector
        final_node_vector = np.zeros(vector_size)

        if semantic_features:
            all_tokens_for_node = [
                token 
                for sentence in semantic_features 
                for token in tokenize(sentence)
            ]
            
            if all_tokens_for_node:
                token_vectors = [
                    w2v_model.wv[token] 
                    for token in all_tokens_for_node 
                    if token in w2v_model.wv
                ]
                if token_vectors:
                    # Calculate the average vector
                    final_node_vector = np.mean(token_vectors, axis=0)

        # Core change: Add the computed vector as a new attribute to the node
        nx_g.nodes[node_id]['semantic_vec'] = final_node_vector
            
    return nx_g # Return the modified graph


# ===================================================================
# 2. Main preprocessing function
# ===================================================================

def preload_dataset_with_semantics(path, semantic_dim=32, max_seq_len=64):
    """
    Refactored preprocessing function that integrates semantic feature generation.
    """
    # Adjust this path based on your directory structure
    path = './data/e3/' + path 
    if os.path.exists(path + '/metadata.json'):
        print(f"Dataset {path} has already been preprocessed, skipping.")
        return
    
    print(f"Starting to preprocess dataset: {path}")

    # --- Step 1: Load all raw NetworkX graphs ---
    print("Loading raw NetworkX graphs...")
    train_nx_gs = [nx.node_link_graph(g) for g in pkl.load(open(path + '/train.pkl', 'rb'))]
    test_nx_gs = [nx.node_link_graph(g) for g in pkl.load(open(path + '/test.pkl', 'rb'))]
    all_nx_gs = train_nx_gs + test_nx_gs

    # --- Step 2: Train a global Word2Vec model ---
    print("Preparing global corpus for Word2Vec...")
    corpus = []
    for g in train_nx_gs:
        for _, data in g.nodes(data=True):          
            for sentence in data.get('semantic_features', []):
                tokens = tokenize(sentence)
                if tokens:
                    corpus.append(tokens)
    
    w2v_model_path = path + '/w2v_model.pkl'
   
    if not os.path.exists(w2v_model_path):
        print("Existing Word2Vec model not found, training a new one...")
        w2v_model = Word2Vec(corpus, vector_size=semantic_dim, window=5, min_count=1, sg=1, workers=4, epochs=20)
        w2v_model.save(w2v_model_path)
        print(f"New model saved to {w2v_model_path}")
    else:
        print(f"Loading existing Word2Vec model from {w2v_model_path}...")
        w2v_model = Word2Vec.load(w2v_model_path)

    # --- Step 3: Prepare positional encodings ---
    positional_encodings = get_positional_encoding(max_seq_len, semantic_dim)

    # --- Step 4: Process all graphs and create the final list of DGL graphs ---
    processed_train_gs = []
    processed_test_gs = []

    # Calculate the global number of node types for one-hot encoding
    max_node_type = 0
    for g in all_nx_gs:
        types = [data['type'] for _, data in g.nodes(data=True)]
        if types:
            max_node_type = max(max_node_type, max(types))
    node_type_dim = max_node_type + 1

    print("Start converting training graphs...")
    for nx_g in tqdm(train_nx_gs):
        # Generate semantic embeddings and add them to the graph
        nx_g_with_features = generate_semantic_embeddings_for_graph(nx_g, w2v_model)
        
        # Create DGL graph, pulling in the new 'semantic_vec' attribute
        dgl_g = dgl.from_networkx(nx_g_with_features, 
                                  node_attrs=['type', 'semantic_vec'],
                                  edge_attrs=['encoding'])
        
        # Create final node features by combining one-hot type and semantic vectors
        node_type_features = F.one_hot(dgl_g.ndata['type'], num_classes=node_type_dim).float()
        semantic_features = dgl_g.ndata['semantic_vec'].float()
        dgl_g.ndata['attr'] = torch.cat([node_type_features, semantic_features], dim=1)
        
        # Use the existing one-hot encoding for edges as the final edge attribute
        dgl_g.edata['attr'] = dgl_g.edata['encoding'].float()

        # Clean up intermediate data to save memory
        del dgl_g.ndata['type'], dgl_g.edata['encoding'], dgl_g.ndata['semantic_vec']
        
        processed_train_gs.append(dgl_g)

    print("Start converting testing graphs...")
    for nx_g in tqdm(test_nx_gs):
        nx_g_with_features = generate_semantic_embeddings_for_graph(nx_g, w2v_model)
        dgl_g = dgl.from_networkx(nx_g_with_features, 
                                  node_attrs=['type', 'semantic_vec'],
                                  edge_attrs=['encoding'])
        
        node_type_features = F.one_hot(dgl_g.ndata['type'], num_classes=node_type_dim).float()
        semantic_features = dgl_g.ndata['semantic_vec'].float()
        dgl_g.ndata['attr'] = torch.cat([node_type_features, semantic_features], dim=1)
        dgl_g.edata['attr'] = dgl_g.edata['encoding'].float()
        
        del dgl_g.ndata['type'], dgl_g.ndata['semantic_vec'], dgl_g.edata['encoding']
        
        processed_test_gs.append(dgl_g)

    # --- Step 5: Load and save metadata and processed graphs ---
    print("Saving processed DGL graphs and metadata...")
    malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))
    train_malicious_nodes = pkl.load(open(path + '/train_malicious_nodes.pkl', 'rb'))
    with open(path + '/malicious_train_list.txt', 'r', encoding='utf-8') as fr:
        malicious_train_list = json.load(fr)

    final_node_feature_dim = processed_train_gs[0].ndata['attr'].shape[1]
    final_edge_feature_dim = processed_train_gs[0].edata['attr'].shape[1]

    metadata = {
        'node_feature_dim': final_node_feature_dim,
        'edge_feature_dim': final_edge_feature_dim,
        'malicious': malicious,
        'malicious_labels': malicious_train_list,
        'train_malicious_nodes': train_malicious_nodes,
        'n_train': len(processed_train_gs),
        'n_test': len(processed_test_gs)
    }
    with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f)

    for i, g in enumerate(processed_train_gs):
        with open(path + '/train_set{}.pkl'.format(i), 'wb') as f:
            pkl.dump(g, f)
    for i, g in enumerate(processed_test_gs):
        with open(path + '/test_set{}.pkl'.format(i), 'wb') as f:
            pkl.dump(g, f)

    print(f"Dataset {path} preprocessing complete!")

# --- Example of how to run the script ---
if __name__ == '__main__':
    # Assuming your dataset folders are 'trace', 'cadets', etc.
    dataset_name = 'cadets' 
    preload_dataset_with_semantics(dataset_name)