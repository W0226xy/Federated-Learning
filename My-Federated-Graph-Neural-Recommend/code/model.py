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
        return (self.user_ids[idx], self.item_ids[idx], self.history[idx], self.neighbor_emb[idx]), self.labels[idx]


class GraphRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, num_heads=4, dropout=0.2):
        super(GraphRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
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

    def forward(self, user_ids, item_ids, history, neighbor_emb):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # History embedding aggregation with multi-head attention
        history_emb = self.item_embedding(history)
        print(f"Original neighbor_emb shape: {neighbor_emb.shape}")
        if neighbor_emb.dim() < 4:
            # Fill missing neighbor embeddings with default values (e.g., zeros)
            neighbor_emb = torch.zeros(user_ids.size(0), history.size(1), 100, history_emb.size(-1), device=history.device)
            print(f"Filled neighbor_emb shape: {neighbor_emb.shape}")

        padding_size = history_emb.size(-1) - neighbor_emb.size(-1)
        if padding_size > 0:
            neighbor_emb = F.pad(neighbor_emb, (0, padding_size), "constant", 0)
        print(f"Adjusted neighbor_emb shape: {neighbor_emb.shape}")

        # Repeat neighbor_emb to match batch size and history length dimensions
        neighbor_emb = neighbor_emb.repeat(history_emb.size(0) // neighbor_emb.size(0), 1, 1, 1)
        print(f"Repeated neighbor_emb shape: {neighbor_emb.shape}")

        # Repeat history_emb to match neighbor_emb dimensions
        history_emb_expanded = history_emb.unsqueeze(2).repeat(1, 1, neighbor_emb.size(2), 1)
        print(f"Expanded history_emb shape: {history_emb_expanded.shape}")

        attention_inputs = torch.cat((history_emb_expanded, neighbor_emb), dim=-1)
        attention_scores = F.relu(self.attention_layer(attention_inputs.view(-1, 2 * history_emb.size(-1))))
        attention_weights = F.softmax(self.attention_mlp(attention_scores).view(neighbor_emb.size(0), neighbor_emb.size(1), neighbor_emb.size(2)), dim=-1)
        aggregated_emb = (attention_weights.unsqueeze(-1) * neighbor_emb).sum(dim=-2)

        # Multi-head projection and aggregation
        multihead_embs = [proj(aggregated_emb) for proj in self.multihead_proj]
        aggregated_emb = torch.cat(multihead_embs, dim=-1) if len(multihead_embs) > 0 else aggregated_emb

        # Reduce sequence dimension of aggregated_emb
        aggregated_emb = aggregated_emb.mean(dim=1)

        # Combine user embedding and aggregated neighbor embedding
        combined_emb = torch.cat([user_emb, aggregated_emb], dim=-1)
        combined_emb = F.relu(self.fc1(combined_emb))
        combined_emb = self.dropout(combined_emb)

        # Final output layer
        output = torch.sigmoid(self.output_layer(combined_emb))
        return output.squeeze()
