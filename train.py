import os
import random
import torch
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train, batch_level_train_with_malicious
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
from model.discriminator import Discriminator
from torch import nn
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size, is_train=True):
    """Creates a DGL GraphDataLoader from a list of dataset indices."""
    if is_train:
        random.shuffle(entries)
    
    sampler = SubsetRandomSampler(torch.arange(len(entries)))
    data_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=sampler)
    return data_loader


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset

    # --- Dataset-specific Hyperparameter Configuration ---
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.max_epoch = 5
        main_args.num_layers = 4
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.max_epoch = 2
        main_args.num_layers = 4
    else: # For entity-level datasets
        main_args.num_hidden = 64
        main_args.max_epoch = 50
        main_args.num_layers = 3
    set_random_seed(0)

    # --- Training Logic for Batch-Level Datasets ---
    if dataset_name in ['streamspot', 'wget']:
        batch_size = 12 if dataset_name == 'streamspot' else 1
        
        dataset = load_batch_level_dataset(dataset_name)
        main_args.n_dim = dataset['n_feat']
        main_args.e_dim = dataset['e_feat']
        graphs = dataset['dataset']

        model = build_model(main_args).to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)

        # Prepare dataloaders for training and contrastive sets
        train_loader = extract_dataloaders(dataset['train_index'], batch_size)
        malicious_loader = extract_dataloaders(dataset['malicious_index'], 1, is_train=False)

        model = batch_level_train_with_malicious(model, graphs, train_loader,
                                                 malicious_loader, optimizer,
                                                 main_args.max_epoch, device,
                                                 main_args.n_dim, main_args.e_dim)
        
        torch.save(model.state_dict(), f"./checkpoints/checkpoint-{dataset_name}.pt")

    # --- Training Logic for Entity-Level Datasets ---
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        
        model = build_model(main_args).to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)

        n_train = metadata['n_train']
        epoch_iter = tqdm(range(main_args.max_epoch), desc="Training")
        
        for epoch in epoch_iter:
            cumulative_loss = 0.0
            
            # The last graph in the training set is consistently used as the negative/contrastive sample
            contrastive_g = load_entity_level_dataset(dataset_name, 'train', n_train - 1).to(device)
            
            # Iterate through all other graphs for training
            for i in range(n_train - 1):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                
                optimizer.zero_grad()
                model.train()
                
                loss = model(g, None, contrastive_g)
                # Note: Loss is scaled before backpropagation
                loss /= (n_train - 1)
                
                loss.backward()
                optimizer.step()
                
                cumulative_loss += loss.item()
                del g

            del contrastive_g
            
            epoch_iter.set_description(f"Epoch {epoch} | Train Loss: {cumulative_loss:.4f}")
            
        torch.save(model.state_dict(), f"./checkpoints/checkpoint-{dataset_name}.pt")
        
        # Clean up previous evaluation results to ensure a fresh run
        save_dict_path = f'./eval_result/distance_save_{dataset_name}.pkl'
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
            
    return


if __name__ == '__main__':
    args = build_args()
    main(args)