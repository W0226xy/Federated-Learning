# client.py

import torch
from torch.utils.data import DataLoader



class Client:
    def __init__(self, client_id, data, model, device, batch_size=32):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.data_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def get_interactions(self):
        """
        按批次返回客户端的交互数据，不使用 DataLoader，而是手动分批。
        """
        num_data = len(self.user_ids)

        # 按批次处理数据
        for start_idx in range(0, num_data, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_data)

            # 获取一个批次的数据
            batch_user_ids = self.user_ids[start_idx:end_idx]
            batch_item_ids = self.item_ids[start_idx:end_idx]
            batch_history = self.history[start_idx:end_idx]
            batch_neighbor_emb = self.neighbor_emb[start_idx:end_idx]
            batch_labels = self.labels[start_idx:end_idx]

            # 返回一个元组，包含用户、物品、历史、邻居嵌入和标签
            yield (batch_user_ids, batch_item_ids, batch_history, batch_neighbor_emb), batch_labels

    def train(self, epochs, lr):
        # 使用本地数据在客户端模型上进行训练
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
        # 获取训练后的模型梯度
        gradients = [param.grad.clone() for param in self.model.parameters()]
        return gradients

    def get_data_size(self):
        # 返回客户端本地数据的大小
        return len(self.data)

    def update_model(self, global_model_params):
        # 更新客户端模型参数为全局模型参数
        for param, global_param in zip(self.model.parameters(), global_model_params):
            param.data = global_param.data.clone()
