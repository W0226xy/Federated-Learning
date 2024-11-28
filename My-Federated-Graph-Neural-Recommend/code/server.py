# server.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import random
from sklearn.cluster import KMeans
from torch.cuda.amp import autocast, GradScaler
import numpy as np


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
            if item_grad is not None:
                print(f"Debug: item_grad size: {item_grad.size()}, returned_items length: {len(returned_items)}")
            else:
                print("Debug: item_grad is None")
            if user_grad is not None:
                print(f"Debug: user_grad size: {user_grad.size()}, returned_users length: {len(returned_users)}")
            else:
                print("Debug: user_grad is None")

            # 小批量处理
            num_batches_item = (len(returned_items) + batch_size - 1) // batch_size  # 计算物品批次数
            num_batches_user = (len(returned_users) + batch_size - 1) // batch_size  # 计算用户批次数

            for batch_idx in range(num_batches_item):
                start_idx_item = batch_idx * batch_size
                end_idx_item = min((batch_idx + 1) * batch_size, len(returned_items))
                batch_items = returned_items[start_idx_item:end_idx_item]
                batch_item_grad = item_grad[start_idx_item:end_idx_item]

                # Update item embeddings
                item_count[batch_items] += weight
                gradient_item[batch_items] += weight * batch_item_grad

            for batch_idx in range(num_batches_user):
                start_idx_user = batch_idx * batch_size
                end_idx_user = min((batch_idx + 1) * batch_size, len(returned_users))
                batch_users = returned_users[start_idx_user:end_idx_user]
                batch_user_grad = user_grad[start_idx_user:end_idx_user]

                # Update user embeddings
                user_count[batch_users] += weight
                gradient_user[batch_users] += weight * batch_user_grad

        # Normalize gradients by counts
        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)

        # Update model parameters
        for param, grad in zip(self.model_user.parameters(), model_grad_user):
            param.data -= self.lr * grad / len(param_list) + self.weight_decay * param.data
        for param, grad in zip(self.model_item.parameters(), model_grad_item):
            param.data -= self.lr * grad / len(param_list) + self.weight_decay * param.data

        # Update item/user embeddings
        with torch.no_grad():
            self.item_emb.weight -= self.lr * gradient_item + self.weight_decay * self.item_emb.weight
            self.user_emb.weight -= self.lr * gradient_user + self.weight_decay * self.user_emb.weight

    def construct_global_graph(self, selected_clients):
        # Randomly select one user and one item embedding from each client
        selected_user_embs = []
        selected_item_embs = []
        for client in selected_clients:
            user_idx = random.randint(0, self.user_emb.weight.size(0) - 1)
            item_idx = random.randint(0, self.item_emb.weight.size(0) - 1)
            selected_user_embs.append(self.user_emb.weight[user_idx].unsqueeze(0))
            selected_item_embs.append(self.item_emb.weight[item_idx].unsqueeze(0))
        # Print information about the data received from clients
        print("\nConstructing Global Graph with Client Data:")
        for client in selected_clients:
            print(f"Client {client.client_id}: User Embedding = {client.model.user_embedding.weight.data.cpu().numpy()}, Item Embedding = {client.model.item_embedding.weight.data.cpu().numpy()}")
        x = torch.cat([torch.cat(selected_user_embs, dim=0), torch.cat(selected_item_embs, dim=0)], dim=0).to(self.device)
        num_users = self.user_emb.weight.shape[0]
        user_item_edges = []
        for client in selected_clients:
            interactions = client.get_interactions()
            for user, item in interactions:
                user_item_edges.extend([[user, num_users + item], [num_users + item, user]])
        if not user_item_edges:
            print("Warning: user_item_edges is empty. Adding self-loops to prevent empty edge index.")
        edge_index = torch.tensor(user_item_edges, dtype=torch.long).t().contiguous().to(self.device)

        return Data(x=x, edge_index=edge_index).to(self.device)

    def global_graph_training(self):
        if len(self.client_list) < 5:
            print("Warning: Not enough clients to sample. Reducing CLIENTS_PER_ROUND to available clients.")
            selected_clients = self.client_list
        else:
            selected_clients = random.sample(self.client_list, min(5, len(self.client_list)))  # 每轮随机选择最多 5 个客户端

        global_graph = self.construct_global_graph(selected_clients)  # 构建全局图
        optimizer = torch.optim.AdamW(self.global_gat.parameters(), lr=self.lr)
        self.global_gat.train()

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
            with autocast():  # 使用自动混合精度加速
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_users)
                    batch_x = global_graph.x[start_idx:end_idx]
                    batch_edge_index = global_graph.edge_index[:, start_idx:end_idx]
                    out = self.global_gat(batch_x, batch_edge_index)  # 前向计算
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
