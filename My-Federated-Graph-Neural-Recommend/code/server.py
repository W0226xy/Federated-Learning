import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.cuda.amp import GradScaler
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

# Server 类
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
        # 聚合客户端更新的模型参数和梯度
        gradient_item = torch.zeros_like(self.item_emb.weight)
        gradient_user = torch.zeros_like(self.user_emb.weight)
        item_count = torch.zeros(self.item_emb.weight.shape[0]).to(self.device)
        user_count = torch.zeros(self.user_emb.weight.shape[0]).to(self.device)

        total_weight = sum(parameter['num_data_points'] for parameter in param_list)

        for parameter in param_list:
            model_grad_user, model_grad_item = parameter['model']
            item_grad, returned_items = parameter['item']
            user_grad, returned_users = parameter['user']
            weight = parameter['num_data_points'] / total_weight

            if item_grad is not None:
                num_batches_item = (len(returned_items) + batch_size - 1) // batch_size  # 计算物品批次数
                for batch_idx in range(num_batches_item):
                    start_idx_item = batch_idx * batch_size
                    end_idx_item = min((batch_idx + 1) * batch_size, len(returned_items))
                    batch_items = returned_items[start_idx_item:end_idx_item]
                    batch_item_grad = item_grad[start_idx_item:end_idx_item]
                    item_count[batch_items] += weight
                    gradient_item[batch_items] += weight * batch_item_grad

            if user_grad is not None:
                num_batches_user = (len(returned_users) + batch_size - 1) // batch_size  # 计算用户批次数
                for batch_idx in range(num_batches_user):
                    start_idx_user = batch_idx * batch_size
                    end_idx_user = min((batch_idx + 1) * batch_size, len(returned_users))
                    batch_users = returned_users[start_idx_user:end_idx_user]
                    batch_user_grad = user_grad[start_idx_user:end_idx_user]
                    user_count[batch_users] += weight
                    gradient_user[batch_users] += weight * batch_user_grad

        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)

        for param, grad in zip(self.model_user.parameters(), model_grad_user):
            param.data -= self.lr * grad / len(param_list) + self.weight_decay * param.data
        for param, grad in zip(self.model_item.parameters(), model_grad_item):
            param.data -= self.lr * grad / len(param_list) + self.weight_decay * param.data

        with torch.no_grad():
            self.item_emb.weight -= self.lr * gradient_item + self.weight_decay * self.item_emb.weight
            self.user_emb.weight -= self.lr * gradient_user + self.weight_decay * self.user_emb.weight



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
                    # 修正错误的节点索引
                    if i < num_users and idx < num_items:
                        edges.append((i, num_users + idx))  # 添加边（用户与物品之间）
                        edge_weights.append(sim)  # 添加相似度作为边的权重
                    else:
                        print(f"Skipping invalid edge ({i}, {num_users + idx}) due to invalid index.")

        # 创建图的数据结构
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置为 (2, num_edges)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)  # 边的权重

        # 检查边索引是否有效
        num_nodes = num_users + num_items
        if (edge_index >= num_nodes).any():
            print(f"Invalid edge_index detected: {edge_index}")
            raise ValueError(
                f"Invalid edge index detected! Ensure that all edge indices are within the valid range of nodes (0 to {num_nodes - 1}).")

        # 将用户和物品嵌入作为节点特征
        x = torch.tensor(np.concatenate([all_user_embeddings, all_item_embeddings], axis=0), dtype=torch.float)

        # 构建最终的图对象
        global_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        print("Graph construction completed.")
        return global_graph

    def global_graph_training(self):
        if len(self.client_list) < 5:
            print("Warning: Not enough clients to sample. Reducing CLIENTS_PER_ROUND to available clients.")
            selected_clients = self.client_list
        else:
            selected_clients = random.sample(self.client_list, min(5, len(self.client_list)))  # 每轮随机选择最多 5 个客户端

        global_graph = self.construct_global_graph(selected_clients)  # 构建全局图
        optimizer = torch.optim.AdamW(self.global_gat.parameters(), lr=self.lr)
        self.global_gat.train()

        # 确保模型在正确的设备上
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_gat = self.global_gat.to(device)  # 将模型移到设备上

        # Early stopping parameters
        patience = 10
        best_loss = np.inf
        patience_counter = 0

        # Mini-batch processing
        batch_size = 64  # 设置批量大小
        num_users = global_graph.x.shape[0]
        num_batches = num_users // batch_size
        if num_users % batch_size != 0:
            num_batches += 1  # 如果不能整除，则增加一个批次

        for epoch in range(100):  # 迭代训练
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # 清除未使用的缓存
            with torch.amp.autocast('cuda'):  # 使用自动混合精度加速
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_users)
                    batch_x = global_graph.x[start_idx:end_idx].to(device)  # 将数据移到同一个设备
                    batch_edge_index = global_graph.edge_index[:, start_idx:end_idx].to(device)  # 同上

                    # 1. 确保 edge_index 在合法范围内
                    num_nodes = batch_x.shape[0]  # batch_x 的第一个维度应该是节点的总数
                    batch_edge_index = map_to_valid_range(batch_edge_index, num_nodes)

                    # 2. 检查 edge_index 是否越界
                    if (batch_edge_index >= num_nodes).any():
                        print(f"Invalid edge_index detected: {batch_edge_index}")
                        raise ValueError(
                            f"Invalid edge index detected! Ensure that all edge indices are within the valid range of nodes (0 to {num_nodes - 1}).")

                    # 前向计算
                    out = self.global_gat(batch_x, batch_edge_index)  # 前向计算

                    # 3. 检查 alpha 是否包含 NaN 或 inf
                    if torch.isnan(out).any():
                        raise ValueError("NaN detected in output.")
                    if torch.isinf(out).any():
                        raise ValueError("Infinity detected in output.")

                    loss = F.mse_loss(out, batch_x)  # 损失计算
                    self.scaler.scale(loss).backward()  # 反向传播
                self.scaler.step(optimizer)  # 更新优化器
                self.scaler.update()  # 更新混合精度

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss}")
                break

        num_users = self.user_emb.weight.shape[0]
        self.user_emb.weight.data = out[:num_users]  # 更新用户嵌入
        self.item_emb.weight.data = out[num_users:]  # 更新物品嵌入


# GlobalGraphGAT 类
class GlobalGraphGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GlobalGraphGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        print(f"x.shape: {x.shape}")  # 查看输入特征的形状
        print(f"edge_index.shape: {edge_index.shape}")  # 查看边的索引形状

        x = F.relu(self.gat1(x, edge_index))  # 第一层 GAT
        print(f"After gat1, x.shape: {x.shape}")  # 查看经过 gat1 后的形状

        x = self.gat2(x, edge_index)  # 第二层 GAT
        print(f"After gat2, x.shape: {x.shape}")  # 查看经过 gat2 后的形状

        return x


def map_to_valid_range(edge_index, num_nodes):
        """
        将超出合法范围的节点索引映射到 [0, num_nodes-1] 范围内。
        """
        # 对每个节点索引应用模运算
        edge_index = edge_index % num_nodes
        return edge_index


