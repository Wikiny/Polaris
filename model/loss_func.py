import torch.nn.functional as F
import torch

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def uniform_loss(self, features, t=2, max_size=30000, batch=10000):
    n = features.size(0)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    if n < max_size:
        dot_product = torch.matmul(features, features.T)
        loss = torch.log(torch.exp(2. * t * (dot_product - 1.)).mean())
    else:
        total_loss = 0.
        permutation = torch.randperm(n, device=features.device)
        features = features[permutation]
        num_batches = (n + batch - 1) // batch
        for i in range(0, n, batch):
            batch_features = features[i:i + batch]
            dot_product = torch.matmul(batch_features, batch_features.T)
            batch_loss = torch.log(torch.exp(2. * t * (dot_product - 1.)).mean())
            total_loss += batch_loss
        loss = total_loss / num_batches
    return loss