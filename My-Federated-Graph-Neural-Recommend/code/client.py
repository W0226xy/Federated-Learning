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
        # Assuming `self.data` contains tuples of (user_id, item_id, label) or (user_id, item_id)
        interactions = []
        for data_point in self.data:
            if len(data_point) == 3:
                user_id, item_id, label = data_point
            elif len(data_point) == 2:
                user_id, item_id = data_point
            else:
                raise ValueError("Unexpected data format in client data")
            interactions.append((user_id, item_id))
        return interactions

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
