# client.py

import torch
from const import HIS_LEN, NEIGHBOR_LEN, HIDDEN
import numpy as np
# 移除未使用的导入
# from expansion import graph_embedding_expansion

class FederatedClient:
    def __init__(self, client_id, local_data, model, device):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # 混合精度梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler()

        print(f"[DEBUG] Client {self.client_id} initialized with local data:")
        # 检查批次数量
        print(f"  Number of batches: {len(self.local_data['batches'])}")
        # 仅打印前两个批次进行调试
        for batch_index, batch in enumerate(self.local_data['batches']):
            if batch_index >= 2:  # 只打印前2个批次以避免混乱
                break
            print(f"  Batch {batch_index + 1}:")
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs
            print(f"    user_ids: {user_ids[:5]}")
            print(f"    item_ids: {item_ids[:5]}")
            print(f"    history: {history[:5]}")
            print(f"    neighbor_emb: {neighbor_emb[:5]}")
            print(f"    labels: {labels[:5]}")

    def train(self, global_model_params):
        self.model.load_state_dict(global_model_params)
        self.model.to(self.device)
        self.model.train()

        print(f"[DEBUG] Client {self.client_id} begins training. Number of batches: {len(self.local_data['batches'])}")

        accumulation_steps = 4  # 梯度累积步数
        epoch_loss = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.local_data['batches']):
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs

            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            history = history.to(self.device)
            labels = labels.to(self.device)
            neighbor_emb = neighbor_emb.to(self.device).float()  # 确保数据类型和设备一致

            # 使用批次中预先计算的邻居嵌入，无需重新生成
            batch_neighbor_emb = neighbor_emb
            # 调试信息：确认 user_ids 和对应的 neighbor_emb
            print(f"[DEBUG] Client {self.client_id} - Batch {batch_idx + 1} - user_ids: {user_ids.cpu().numpy()}")
            print(
                f"[DEBUG] Client {self.client_id} - Batch {batch_idx + 1} - neighbor_emb shape: {batch_neighbor_emb.shape}")
            print(
                f"[DEBUG] Client {self.client_id} - Batch {batch_idx + 1} - neighbor_emb sample: {batch_neighbor_emb[0][0][0][:5]}")
            # 调整维度（如果需要）
            if batch_neighbor_emb.shape[2] > NEIGHBOR_LEN:
                batch_neighbor_emb = batch_neighbor_emb[:, :, :NEIGHBOR_LEN, :]
            elif batch_neighbor_emb.shape[2] < NEIGHBOR_LEN:
                pad_size = NEIGHBOR_LEN - batch_neighbor_emb.shape[2]
                pad = torch.zeros(
                    batch_neighbor_emb.shape[0], batch_neighbor_emb.shape[1], pad_size, batch_neighbor_emb.shape[3],
                    device=self.device
                )
                batch_neighbor_emb = torch.cat((batch_neighbor_emb, pad), dim=2)

            # 使用混合精度进行前向传播
            with torch.cuda.amp.autocast():
                output = self.model(user_ids, item_ids, history, batch_neighbor_emb)
                loss = torch.nn.functional.mse_loss(output, labels)
                loss = loss / accumulation_steps  # 归一化损失以进行累积

            # 每10个批次打印一次实际评分和预测评分以进行调试
            if batch_idx % 10 == 0:
                print(f"[Client {self.client_id}] Batch {batch_idx + 1}/{len(self.local_data['batches'])}")
                print(f"真实评分 (labels): {labels.cpu().numpy()[:5]}")  # 仅显示前5个以简化
                print(f"预测评分 (output): {output.cpu().detach().numpy()[:5]}")  # 仅显示前5个以简化

            # 使用梯度缩放进行反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # 打印批次损失
            print(f"[Client {self.client_id}] Batch {batch_idx + 1}/{len(self.local_data['batches'])}, Loss: {loss.item() * accumulation_steps}")
            epoch_loss += loss.item() * accumulation_steps

        # 处理剩余的梯度（如果有）
        if len(self.local_data['batches']) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        print(f"[Client {self.client_id}] Epoch Loss: {epoch_loss / len(self.local_data['batches'])}")
        gradients = [
            param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in self.model.parameters()
        ]
        return gradients
