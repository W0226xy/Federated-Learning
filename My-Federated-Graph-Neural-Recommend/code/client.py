import torch


class Client:
    def __init__(self, client_id, data, model, device):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.device = device

    def train(self, epochs, lr):
        # 使用本地数据在客户端模型上进行训练
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            for user_ids, item_ids, labels in self.data:
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