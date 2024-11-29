# server.py
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import random
from sklearn.cluster import KMeans
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import networkx as nx

class Server(nn.Module):
    def __init__(self, client_list, model, user_features, item_features, args):
        super().__init__()
        self.device = args.device
        self.client_list = client_list
        self.model_user, self.model_item = model
        self.user_emb = nn.Embedding.from_pretrained(torch.Tensor(user_features), freeze=False).to(self.device)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(item_features), freeze=False).to(self.device)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.global_gat = GlobalGraphGAT(user_features.shape[1], 8, user_features.shape[1]).to(self.device)
        self.prev_gradient_item = torch.zeros_like(self.item_emb.weight)
        self.prev_gradient_user = torch.zeros_like(self.user_emb.weight)
        self.scaler = GradScaler()  # Mixed Precision Training Scaler

    def aggregate(self, param_list, batch_size=64):
        gradient_item = torch.zeros_like(self.item_emb.weight)
        gradient_user = torch.zeros_like(self.user_emb.weight)
        item_count = torch.zeros(self.item_emb.weight.shape[0]).to(self.device)
        user_count = torch.zeros(self.user_emb.weight.shape[0]).to(self.device)

        total_weight = sum(parameter['num_data_points'] for parameter in param_list)

        # 循环遍历每个客户端的参数，并按照批次进行梯度计算
        for parameter in param_list:
            model_grad_user, model_grad_item = parameter['model']
            item_grad, returned_items = parameter['item']
            user_grad, returned_users = parameter['user']
            weight = parameter['num_data_points'] / total_weight

            # Debugging information
            print(f"Debug: item_grad = {item_grad}, returned_items = {returned_items}")
            print(f"Debug: user_grad = {user_grad}, returned_users = {returned_users}")

            # 检查 item_grad 和 user_grad 是否为 None
            if item_grad is not None:
                num_batches_item = (len(returned_items) + batch_size - 1) // batch_size  # 计算物品批次数
                for batch_idx in range(num_batches_item):
                    start_idx_item = batch_idx * batch_size
                    end_idx_item = min((batch_idx + 1) * batch_size, len(returned_items))
                    batch_items = returned_items[start_idx_item:end_idx_item]
                    batch_item_grad = item_grad[start_idx_item:end_idx_item]

                    # Update item embeddings
                    item_count[batch_items] += weight
                    gradient_item[batch_items] += weight * batch_item_grad
            else:
                print("Warning: item_grad is None, skipping gradient update for items.")

            if user_grad is not None:
                num_batches_user = (len(returned_users) + batch_size - 1) // batch_size  # 计算用户批次数
                for batch_idx in range(num_batches_user):
                    start_idx_user = batch_idx * batch_size
                    end_idx_user = min((batch_idx + 1) * batch_size, len(returned_users))
                    batch_users = returned_users[start_idx_user:end_idx_user]
                    batch_user_grad = user_grad[start_idx_user:end_idx_user]

                    # Update user embeddings
                    user_count[batch_users] += weight
                    gradient_user[batch_users] += weight * batch_user_grad
            else:
                print("Warning: user_grad is None, skipping gradient update for users.")

        # Normalize gradients by counts
        item_count[item_count == 0] = 1  # 防止除以零
        user_count[user_count == 0] = 1  # 防止除以零
        gradient_item /= item_count.unsqueeze(1)  # 按照每个用户的数量进行归一化
        gradient_user /= user_count.unsqueeze(1)  # 按照每个物品的数量进行归一化

        # Update model parameters
        for param, grad in zip(self.model_user.parameters(), model_grad_user):
            param.data -= self.lr * grad / len(param_list) + self.weight_decay * param.data
        for param, grad in zip(self.model_item.parameters(), model_grad_item):
            param.data -= self.lr * grad / len(param_list) + self.weight_decay * param.data

        # Update item/user embeddings
        with torch.no_grad():
            self.item_emb.weight -= self.lr * gradient_item + self.weight_decay * self.item_emb.weight
            self.user_emb.weight -= self.lr * gradient_user + self.weight_decay * self.user_emb.weight

    import networkx as nx

    def cosine_similarity(a, b):
        """
        计算两个向量 a 和 b 之间的余弦相似度
        """
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()

    import torch
    import numpy as np
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors

    import torch
    from torch_geometric.data import Data

    def construct_global_graph(self, selected_clients, k_neighbors=10):
        """
        使用用户嵌入和物品嵌入构建全局图，使用 KNN 查找每个用户的最相似的物品
        :param selected_clients: 被选中的客户端列表
        :param k_neighbors: 每个用户的 K 个最近邻物品数量
        :return: 构建的全局图 (torch_geometric.data.Data)
        """
        # 创建空图
        all_user_embeddings = []
        all_item_embeddings = []

        for client in selected_clients:
            # 获取当前客户端的用户和物品嵌入
            user_embeddings = client.model.user_embedding.weight.data.cpu().numpy()
            item_embeddings = client.model.item_embedding.weight.data.cpu().numpy()

            # 将这些嵌入添加到全局嵌入列表
            all_user_embeddings.append(user_embeddings)
            all_item_embeddings.append(item_embeddings)

        # 将所有客户端的嵌入整合成一个单一的矩阵
        all_user_embeddings = np.concatenate(all_user_embeddings, axis=0)  # (num_users, embedding_dim)
        all_item_embeddings = np.concatenate(all_item_embeddings, axis=0)  # (num_items, embedding_dim)

        # 添加用户节点和物品节点到图中
        num_users = all_user_embeddings.shape[0]
        num_items = all_item_embeddings.shape[0]

        # 创建图的边和相似度
        edges = []
        edge_weights = []
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        knn.fit(all_item_embeddings)  # 使用物品嵌入来训练 KNN 模型

        for i in range(num_users):
            user_embedding = torch.tensor(all_user_embeddings[i]).unsqueeze(0)  # (1, embedding_dim)

            # 找到与当前用户最相似的 K 个物品
            distances, indices = knn.kneighbors(user_embedding.numpy())  # 获取与用户最相似的物品索引和距离
            for idx, dist in zip(indices[0], distances[0]):
                sim = 1 - dist  # 将距离转化为相似度（相似度 = 1 - 距离）
                if sim > 0:
                    edges.append((i, num_users + idx))  # 添加边（用户与物品之间）
                    edge_weights.append(sim)  # 添加相似度作为边的权重

        # 创建图的数据结构
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置为 (2, num_edges)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)  # 边的权重

        # 将用户和物品嵌入作为节点特征
        x = torch.tensor(np.concatenate([all_user_embeddings, all_item_embeddings], axis=0), dtype=torch.float)

        # 构建最终的图对象
        global_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        print("Graph construction completed.")
        return global_graph

    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    def global_graph_training(self):
        # 确保客户端数量足够
        if len(self.client_list) < 5:
            print("Warning: Not enough clients to sample. Reducing CLIENTS_PER_ROUND to available clients.")
            selected_clients = self.client_list
        else:
            selected_clients = random.sample(self.client_list, min(5, len(self.client_list)))  # 每轮随机选择最多5个客户端

        # 构建全局图
        global_graph = self.construct_global_graph(selected_clients)
        optimizer = torch.optim.AdamW(self.global_gat.parameters(), lr=self.lr)
        self.global_gat.train()

        # 初始化混合精度训练
        self.scaler = GradScaler()  # 初始化混合精度训练的Scaler
        patience = 10
        best_loss = float('inf')
        patience_counter = 0

        # Mini-batch 处理
        batch_size = 64  # 设置批量大小
        num_users = global_graph.x.shape[0]  # 获取节点数量
        num_batches = num_users // batch_size
        if num_users % batch_size != 0:
            num_batches += 1  # 如果不能整除，增加一个批次

        # 训练100个epoch
        for epoch in range(100):
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # 清理显存

            # 开启自动混合精度训练
            with autocast():  # 使用自动混合精度加速
                epoch_loss = 0  # 初始化当前epoch的总损失
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_users)

                    # 提取当前批次的数据
                    batch_x = global_graph.x[start_idx:end_idx]
                    batch_edge_index = global_graph.edge_index[:, start_idx:end_idx]

                    # 检查数据形状
                    print(f"Batch {batch_idx + 1}/{num_batches}")
                    print(f"batch_x shape: {batch_x.shape}")
                    print(f"batch_edge_index shape: {batch_edge_index.shape}")

                    # 前向传播
                    out = self.global_gat(batch_x, batch_edge_index)

                    # 损失计算，假设目标是节点特征本身（自监督任务）
                    loss = F.mse_loss(out, batch_x)
                    epoch_loss += loss.item()  # 累加损失

                    # 反向传播
                    self.scaler.scale(loss).backward()  # 使用混合精度反向传播

                # 更新参数
                self.scaler.step(optimizer)  # 更新优化器
                self.scaler.update()  # 更新混合精度

            # 输出当前epoch的损失
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / num_batches:.4f}")

            # 提前停止检查
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss}")
                break

        # 更新用户和物品嵌入
        num_users = self.user_emb.weight.shape[0]  # 获取用户数量
        self.user_emb.weight.data = out[:num_users]  # 更新用户嵌入
        self.item_emb.weight.data = out[num_users:]  # 更新物品嵌入

        # 输出最后更新的嵌入
        print(f"Updated user embeddings shape: {self.user_emb.weight.shape}")
        print(f"Updated item embeddings shape: {self.item_emb.weight.shape}")

    def local_fine_tuning(self, client):
        # 每个客户端对全局模型进行本地微调
        local_model_user = self.model_user
        local_model_item = self.model_item
        optimizer = torch.optim.AdamW(list(local_model_user.parameters()) + list(local_model_item.parameters()), lr=self.lr)
        local_model_user.train()
        local_model_item.train()
        for epoch in range(10):  # 假设每个客户端进行 10 轮本地训练
            for user_ids, item_ids, labels in client.get_local_data():
                optimizer.zero_grad()
                output_user = local_model_user(user_ids, item_ids)
                output_item = local_model_item(user_ids, item_ids)
                loss_user = F.mse_loss(output_user, labels)
                loss_item = F.mse_loss(output_item, labels)
                loss = (loss_user + loss_item) / 2
                loss.backward()
                optimizer.step()


class GlobalGraphGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GlobalGraphGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)
