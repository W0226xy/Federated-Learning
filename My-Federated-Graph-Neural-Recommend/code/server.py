import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import random
from sklearn.neighbors import NearestNeighbors
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

    def aggregate(self, param_list):
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

            # Debugging information
            print(f"Debug: item_grad size: {item_grad.size()}, returned_items length: {len(returned_items)}")
            print(f"Debug: user_grad size: {user_grad.size()}, returned_users length: {len(returned_users)}")

            if item_grad.size(0) != len(returned_items):
                print("Warning: item_grad size and returned_items length do not match. Padding item_grad.")
                padding_size = abs(len(returned_items) - item_grad.size(0))
                if item_grad.size(0) < len(returned_items):
                    mean_value = item_grad.mean(dim=0)
                    padding_tensor = mean_value.unsqueeze(0).repeat(padding_size, 1)
                    item_grad = torch.cat((item_grad, padding_tensor), dim=0)
                else:
                    item_grad = item_grad[:len(returned_items)]

            if user_grad.size(0) != len(returned_users):
                print("Warning: user_grad size and returned_users length do not match. Padding user_grad.")
                padding_size = abs(len(returned_users) - user_grad.size(0))
                if user_grad.size(0) < len(returned_users):
                    mean_value = user_grad.mean(dim=0)
                    padding_tensor = mean_value.unsqueeze(0).repeat(padding_size, 1)
                    user_grad = torch.cat((user_grad, padding_tensor), dim=0)
                else:
                    user_grad = user_grad[:len(returned_users)]

            item_count[returned_items] += weight
            user_count[returned_users] += weight

            if gradient_item.size(1) != item_grad.size(1):
                item_grad = F.pad(item_grad, (0, gradient_item.size(1) - item_grad.size(1)), 'constant', 0)
            gradient_item[returned_items] += weight * item_grad

            if gradient_user.size(1) != user_grad.size(1):
                user_grad = F.pad(user_grad, (0, gradient_user.size(1) - user_grad.size(1)), 'constant', 0)
            gradient_user[returned_users] += weight * user_grad

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
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0).to(self.device)
        num_users = self.user_emb.weight.shape[0]
        user_item_edges = []
        for client in selected_clients:
            interactions = client.get_interactions()
            for user, item in interactions:
                user_item_edges.extend([[user, num_users + item], [num_users + item, user]])
        if not user_item_edges:
            print("Warning: user_item_edges is empty. Adding self-loops to prevent empty edge index.")
        edge_index = torch.tensor(user_item_edges, dtype=torch.long).t().contiguous().to(self.device)

        # 使用 KNN 方法，根据节点特征的相似性来添加额外的边，使用距离阈值过滤邻居
        knn = NearestNeighbors(n_neighbors=10, metric='cosine').fit(x.cpu().detach().numpy())
        knn_distances, knn_indices = knn.kneighbors(x.cpu().detach().numpy())

        distance_threshold = 0.3  # 设置一个距离阈值
        additional_edges = []
        for idx, (neighbors, distances) in enumerate(zip(knn_indices, knn_distances)):
            for neighbor, distance in zip(neighbors, distances):
                if idx != neighbor and distance < distance_threshold:  # Prevent self-loop and use threshold
                    additional_edges.append([idx, neighbor])
                    additional_edges.append([neighbor, idx])
        if additional_edges:
            additional_edge_index = torch.tensor(additional_edges, dtype=torch.long).t().contiguous().to(self.device)
            edge_index = torch.cat([edge_index, additional_edge_index], dim=1)

        return Data(x=x, edge_index=edge_index).to(self.device)

    def global_graph_training(self):
        if len(self.client_list) < 5:
            print("Warning: Not enough clients to sample. Reducing CLIENTS_PER_ROUND to available clients.")
            selected_clients = self.client_list
        else:
            selected_clients = random.sample(self.client_list, min(5, len(self.client_list)))  # 每轮随机选择最多 5 个客户端
        global_graph = self.construct_global_graph(selected_clients)
        optimizer = torch.optim.AdamW(self.global_gat.parameters(), lr=self.lr)
        self.global_gat.train()

        # Early stopping parameters
        patience = 10
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(100):
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear unused cache
            with autocast():
                out = self.global_gat(global_graph.x, global_graph.edge_index)
                loss = F.mse_loss(out, global_graph.x)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
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
        self.user_emb.weight.data = out[:num_users]
        self.item_emb.weight.data = out[num_users:]

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

    def distribute(self, client_list):
        for client in client_list:
            client.update(self.model_user, self.model_item, self.user_emb.weight, self.item_emb.weight)
            # 每个客户端进行本地微调
            self.local_fine_tuning(client)


class GlobalGraphGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GlobalGraphGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)