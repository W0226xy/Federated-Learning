import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels, history, neighbor_emb):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels
        self.history = history
        self.neighbor_emb = neighbor_emb

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        max_idx = min(len(self.user_ids), len(self.item_ids), len(self.history), len(self.neighbor_emb), len(self.labels))
        if idx >= max_idx:
            idx = max_idx - 1

        return (self.user_ids[idx], self.item_ids[idx], self.history[idx], self.neighbor_emb[idx]), self.labels[idx]

class GraphRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, num_heads=4, dropout=0.2):
        super(GraphRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention_layer = nn.Linear(2 * hidden_dim, 96)
        self.attention_mlp = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.num_heads = num_heads
        self.multihead_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim // num_heads) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # GAT Layers for real interaction data and neighbor aggregation
        self.real_interaction_gat = nn.Linear(2 * hidden_dim, 1)
        self.neighbor_gat = nn.Linear(2 * hidden_dim, 1)

    def forward(self, user_ids, item_ids, history=None, neighbor_emb=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 获取用户和物品的嵌入
        # Ensure user_ids and item_ids are on the same device as the model
        user_ids = user_ids.to(self.user_embedding.weight.device)
        item_ids = item_ids.to(self.item_embedding.weight.device)

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        if history is not None and neighbor_emb is not None:
            # 计算历史嵌入
            history_emb = self.item_embedding(history).to(device)
            history_emb_expanded = history_emb.unsqueeze(2).expand(-1, -1, history_emb.size(1), -1)
            real_attention_inputs = torch.cat(
                (history_emb_expanded, history_emb.unsqueeze(2).expand(-1, -1, history_emb.size(1), -1)), dim=-1)
            real_attention_scores = F.relu(
                self.real_interaction_gat(real_attention_inputs.view(-1, 2 * history_emb.size(-1)))).to(device)
            real_attention_weights = F.softmax(
                real_attention_scores.view(history_emb.size(0), history_emb.size(1), history_emb.size(1)), dim=-1)
            aggregated_real_history_emb = (real_attention_weights.unsqueeze(-1) * history_emb.unsqueeze(2)).sum(dim=-2)

            if neighbor_emb.dim() < 4:
                mean_value = history_emb.mean(dim=1)
                neighbor_emb = mean_value.unsqueeze(1).unsqueeze(2).expand(user_ids.size(0), history.size(1), 100, history_emb.size(-1))

            padding_size = history_emb.size(-1) - neighbor_emb.size(-1)
            if padding_size > 0:
                neighbor_emb = F.pad(neighbor_emb, (0, padding_size), "constant", 0)

            # 邻居聚合
            neighbor_attention_inputs = torch.cat(
                (aggregated_real_history_emb.unsqueeze(2).expand(-1, -1, neighbor_emb.size(2), -1), neighbor_emb),
                dim=-1)
            neighbor_attention_scores = F.relu(
                self.neighbor_gat(neighbor_attention_inputs.view(-1, 2 * history_emb.size(-1)))).to(device)
            neighbor_attention_weights = F.softmax(
                neighbor_attention_scores.view(neighbor_emb.size(0), neighbor_emb.size(1), neighbor_emb.size(2)),
                dim=-1)
            aggregated_emb_real = (neighbor_attention_weights.unsqueeze(-1) * neighbor_emb).sum(dim=-2)

            multihead_embs_real = [proj(aggregated_emb_real) for proj in self.multihead_proj]
            aggregated_emb_real = torch.cat(multihead_embs_real, dim=-1) if len(
                multihead_embs_real) > 0 else aggregated_emb_real

            aggregated_emb = aggregated_real_history_emb.mean(dim=1) + aggregated_emb_real.mean(dim=1)
            combined_emb = torch.cat([user_emb, aggregated_emb, item_emb], dim=-1)
        else:
            combined_emb = torch.cat([user_emb, item_emb], dim=-1)

        combined_emb = F.relu(self.fc1(combined_emb)).to(device)
        combined_emb = self.dropout(combined_emb)
        combined_emb = F.relu(self.fc2(combined_emb)).to(device)
        output = torch.sigmoid(self.output_layer(combined_emb)).to(device)

        return output.squeeze()
