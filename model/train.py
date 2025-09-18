from cProfile import label
import dgl
import numpy as np
from tqdm import tqdm
from utils.loaddata import transform_graph


def batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, n_dim=0, e_dim=0):
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for _, batch in enumerate(train_loader):
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]
            batch_g = dgl.batch(batch_g)
            model.train()
            loss = model(batch_g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model

def batch_level_train_with_malicious(model, graphs, train_loader, malicious_loader,
                                     optimizer, max_epoch, device, n_dim=0, e_dim=0):
    # 初始化一个空列表来存储所有恶意图
    malicious_graphs = []

    # 遍历 malicious_loader 获取所有图
    for _, batch in enumerate(malicious_loader):
        # 假设 loader 返回的是索引，graphs 是原始图数据集
        batch_graphs = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]
        malicious_graphs.extend(batch_graphs)

    # 合并所有恶意图为一个大图
    batched_malicious_graph = dgl.batch(malicious_graphs)
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for _, batch in enumerate(train_loader):
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]
            batch_g = dgl.batch(batch_g)
            model.train()
            loss = model(batch_g,label=None,dif_g=batched_malicious_graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model