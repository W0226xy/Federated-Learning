import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
class Client:
    def __init__(self, client_id, data, model, device, batch_size=32, initial_embedding_mix_ratio=0.5):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.embedding_mix_ratio = initial_embedding_mix_ratio  # 初始嵌入比例
        self.data_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def get_interactions(self):
        """
        按批次返回客户端的交互数据
        """
        num_data = len(self.data)

        # 按批次处理数据
        for start_idx in range(0, num_data, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_data)

            # 获取一个批次的数据
            batch_data = self.data[start_idx:end_idx]
            batch_user_ids = batch_data['user_ids']
            batch_item_ids = batch_data['item_ids']
            batch_history = batch_data['history']
            batch_neighbor_emb = batch_data['neighbor_emb']
            batch_labels = batch_data['labels']

            # 返回用户、物品、历史、邻居嵌入和标签
            yield (batch_user_ids, batch_item_ids, batch_history, batch_neighbor_emb), batch_labels

    def train(self, epochs, lr):
        """
        使用本地数据在客户端模型上进行训练
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            for batch in self.data_loader:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(user_ids, item_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def get_model_gradients(self):
        """
        获取训练后的模型梯度
        """
        gradients = [param.grad.clone() for param in self.model.parameters()]
        return gradients

    def get_data_size(self):
        """
        返回客户端本地数据的大小
        """
        return len(self.data)

    def update_model(self, global_model_params):
        """
        更新客户端模型参数为全局模型参数
        """
        for param, global_param in zip(self.model.parameters(), global_model_params):
            param.data = global_param.data.clone()

    def update_embeddings(self, global_user_embeddings, global_item_embeddings):
        """
        Update the client's embeddings by dynamically adjusting the weight between
        local embeddings and global embeddings using cosine similarity.
        """
        # 获取本地用户嵌入和物品嵌入
        local_user_embeddings = self.model.user_embedding.weight.data.clone()  # 本地用户嵌入
        local_item_embeddings = self.model.item_embedding.weight.data.clone()  # 本地物品嵌入

        # 计算本地嵌入和全局嵌入的余弦相似度
        user_cosine_similarity = torch.cosine_similarity(local_user_embeddings, global_user_embeddings)
        item_cosine_similarity = torch.cosine_similarity(local_item_embeddings, global_item_embeddings)

        # 动态调整比例，这里简单地使用平均值来作为嵌入的加权比例
        user_weight = user_cosine_similarity.mean().item()
        item_weight = item_cosine_similarity.mean().item()

        # 保证权重在 [0, 1] 的范围内
        user_weight = max(0, min(1, user_weight))
        item_weight = max(0, min(1, item_weight))

        # 更新本地嵌入：加权本地和全局嵌入
        self.model.user_embedding.weight.data = user_weight * global_user_embeddings + (
                    1 - user_weight) * local_user_embeddings
        self.model.item_embedding.weight.data = item_weight * global_item_embeddings + (
                    1 - item_weight) * local_item_embeddings

        print(f"Updated user embeddings with weight: {user_weight:.4f}")
        print(f"Updated item embeddings with weight: {item_weight:.4f}")

    def adjust_embedding_mix_ratio(self, new_ratio):
        """
        动态调整嵌入混合比例
        :param new_ratio: 新的嵌入混合比例
        """
        self.embedding_mix_ratio = new_ratio
        print(f"Client {self.client_id} embedding mix ratio updated to {self.embedding_mix_ratio:.2f}")
