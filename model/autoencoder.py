from matplotlib.backend_bases import ToolContainerBase
from .gat import GAT
from utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss,uniform_loss
import torch
import torch.nn as nn
import dgl
import random
from torch.nn import functional as F

def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = SGMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    return model


class SGMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(SGMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        self.supervised_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.triplet_cosine_margin = 0
        self.triplet_cosine_loss = nn.TripletMarginWithDistanceLoss()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.classifier.apply(init_weights)
        
        weighting_net_input_dim = hidden_dim * n_layers * 2
        self.weighting_net = nn.Sequential(
            nn.Linear(weighting_net_input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.weighting_net.apply(init_weights)

        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )

        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)
        self.uniformity_layer = nn.Linear(dec_in_dim * n_layers, dec_in_dim//4, bias=False)
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token
        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g, label, contrastive_g):
        loss = self.compute_loss(g, label, contrastive_g)
        return loss

    def compute_loss(self, g, label=None, contrastive_g=None):
        triplet_loss = torch.tensor(0.0, device=g.device)
        weight_guidance_loss = torch.tensor(0.0, device=g.device)

        # --- Feature Reconstruction ---
        masked_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        masked_node_features = masked_g.ndata['attr'].to(masked_g.device)
        
        _, all_hidden_layers = self.encoder(masked_g, masked_node_features, return_hidden=True)
        masked_node_embeddings = torch.cat(all_hidden_layers, dim=1)
        
        decoder_input_features = self.encoder_to_decoder(masked_node_embeddings)
        reconstructed_features = self.decoder(masked_g, decoder_input_features)

        original_masked_features = g.ndata['attr'][mask_nodes]
        reconstructed_masked_features = reconstructed_features[mask_nodes]
        feature_recon_loss = self.criterion(reconstructed_masked_features, original_masked_features)

        # --- Structural Reconstruction ---
        threshold = min(10000, g.num_nodes())
        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_indices = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_indices], g.edges()[1][positive_edge_indices])

        all_edge_src_indices = torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])
        all_edge_dst_indices = torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])
        
        sample_src_embeddings = masked_node_embeddings[all_edge_src_indices].to(g.device)
        sample_dst_embeddings = masked_node_embeddings[all_edge_dst_indices].to(g.device)
        
        edge_recon_predictions = self.edge_recon_fc(torch.cat([sample_src_embeddings, sample_dst_embeddings], dim=-1)).squeeze(-1)
        edge_recon_labels = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(g.device)
        
        structural_recon_loss = self.recon_loss(edge_recon_predictions, edge_recon_labels)

        # --- Triplet-based Contrastive Loss ---
        if contrastive_g is not None:
            contrastive_g.ndata['attr'] = contrastive_g.ndata['attr'].to(g.device)
            _, all_hidden_negative = self.encoder(contrastive_g, contrastive_g.ndata['attr'], return_hidden=True)
            negative_node_embeddings = torch.cat(all_hidden_negative, dim=1)
            
            _, all_hidden_anchor = self.encoder(g, g.ndata['attr'], return_hidden=True)
            anchor_node_embeddings = torch.cat(all_hidden_anchor, dim=1)
            
            num_anchor_nodes = anchor_node_embeddings.size(0)
            num_negative_nodes = negative_node_embeddings.size(0)

            if num_anchor_nodes > 1 and num_negative_nodes > 0:
                sample_size = 1024
                
                actual_anchor_sample_size = min(sample_size, num_anchor_nodes)
                actual_negative_sample_size = min(sample_size, num_negative_nodes)

                # Anchor samples (from the original graph)
                anchor_indices = torch.randperm(num_anchor_nodes, device=g.device)[:actual_anchor_sample_size]
                anchor_embeds = anchor_node_embeddings[anchor_indices]

                # Positive samples (from the original graph)
                positive_indices = torch.randperm(num_anchor_nodes, device=g.device)[:actual_anchor_sample_size]
                positive_embeds = anchor_node_embeddings[positive_indices]
                
                # Negative samples (from the contrastive graph)
                negative_indices = torch.randperm(num_negative_nodes, device=g.device)[:actual_negative_sample_size]
                negative_embeds = negative_node_embeddings[negative_indices]

                final_triplet_batch_size = min(len(anchor_embeds), len(positive_embeds), len(negative_embeds))

                if final_triplet_batch_size > 0:
                    anchor_final = anchor_embeds[:final_triplet_batch_size]
                    positive_final = positive_embeds[:final_triplet_batch_size]
                    negative_final = negative_embeds[:final_triplet_batch_size]
                    
                    sim_ap = F.cosine_similarity(anchor_final, positive_final, dim=-1)
                    sim_an = F.cosine_similarity(anchor_final, negative_final, dim=-1)
                    
                    losses = torch.relu(sim_an - sim_ap + self.triplet_cosine_margin)
                    
                    weighting_net_input = torch.cat([anchor_final.detach(), negative_final.detach()], dim=-1)
                    difficulty_weights = self.weighting_net(weighting_net_input)
                    
                    epsilon = 1e-8
                    normalized_weights = difficulty_weights / (torch.mean(difficulty_weights) + epsilon)

                    weighted_losses = losses * normalized_weights.squeeze(-1)
                    weight_guidance_loss = -torch.mean(difficulty_weights.squeeze(-1) * sim_an.detach())
                    triplet_loss = weighted_losses.mean()

        weight_guidance_lambda = 0.5
        total_loss = feature_recon_loss + structural_recon_loss + triplet_loss + weight_guidance_loss * weight_guidance_lambda
        
        print(f"FeatureReconLoss: {feature_recon_loss.item():.4f}, StructuralReconLoss: {structural_recon_loss.item():.4f}, TripletLoss: {triplet_loss.item():.4f}, WeightGuidanceLoss: {weight_guidance_loss.item():.4f}")
        return total_loss
        
    def embed(self, g):
        node_features = g.ndata['attr'].to(g.device)
        node_embeddings = self.encoder(g, node_features)
        return node_embeddings

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])