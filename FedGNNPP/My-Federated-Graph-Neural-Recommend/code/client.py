import torch

class FederatedClient:
    def __init__(self, client_id, local_data, model, device):
        self.client_id = client_id
        self.local_data = local_data  # 客户端本地数据
        self.model = model  # 本地模型副本
        self.device = device  # 使用的设备 (CPU/GPU)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, global_model_params):
        """
        使用全局模型参数进行本地训练，训练逻辑保持与模型一致。
        """
        # 加载全局模型参数
        self.model.load_state_dict(global_model_params)
        self.model.to(self.device)
        self.model.train()

        # 本地训练循环
        for user_ids, item_ids, history, neighbor_emb, labels in self.local_data['batches']:
            self.optimizer.zero_grad()

            # 数据转移到设备
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            history = history.to(self.device)
            neighbor_emb = neighbor_emb.to(self.device)
            labels = labels.to(self.device)
            print(f"[DEBUG] Training data - History shape: {history.shape}, Neighbor_emb shape: {neighbor_emb.shape}")
            print(f"[DEBUG] Sample History: {history[:1]}")  # 打印一个用户的历史数据
            print(f"[DEBUG] Sample Neighbor_emb: {neighbor_emb[:1][:1]}")  # 打印一个用户的第一个邻居嵌入

            # 前向传播与损失计算
            output = self.model(user_ids, item_ids, history, neighbor_emb)
            loss = torch.nn.functional.mse_loss(output, labels)

            # 反向传播与优化
            loss.backward()
            self.optimizer.step()

        # 返回更新后的梯度
        gradients = [param.grad.clone() for param in self.model.parameters()]
        return gradients
