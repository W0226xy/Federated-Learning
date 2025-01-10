# client.py

import torch
from const import HIS_LEN, NEIGHBOR_LEN, HIDDEN
import numpy as np
from expansion import graph_embedding_expansion


class FederatedClient:
    def __init__(self, client_id, local_data, model, device, user_neighbor_emb):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Mixed precision gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Store user_neighbor_emb for later use
        self.user_neighbor_emb = user_neighbor_emb

        print(f"[DEBUG] Client {self.client_id} initialized with local data:")
        # Check the number of batches
        print(f"  Number of batches: {len(self.local_data['batches'])}")
        # Iterate over the first few batches for debugging
        for batch_index, batch in enumerate(self.local_data['batches']):
            if batch_index >= 2:  # Only print first 2 batches to avoid clutter
                break
            print(f"  Batch {batch_index + 1}:")
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs
            print(f"    user_ids: {user_ids[:5]}")
            print(f"    item_ids: {item_ids[:5]}")
            print(f"    history: {history[:5]}")
            print(f"    neighbor_emb: {neighbor_emb[:5]}")
            print(f"    labels: {labels[:5]}")

    def train(self, global_model_params, Otraining, usernei, global_embedding):
        """
        针对联邦学习中客户端本地训练的函数，确保在梯度累积的场景下真正执行 optimizer.step() 更新参数。
        """

        # 1) 加载来自服务器的全局模型参数
        self.model.load_state_dict(global_model_params)
        self.model.to(self.device)
        self.model.train()

        print(f"[DEBUG] Client {self.client_id} begins training. "
              f"Number of batches: {len(self.local_data['batches'])}")

        # 如果要使用梯度累积
        accumulation_steps = 4
        epoch_loss = 0.0

        # 2) 重置本地优化器及混合精度的 GradScaler 状态
        self.optimizer.zero_grad()
        self.scaler = torch.cuda.amp.GradScaler()

        # 记录已处理的 batch 数，用于最后一次强制 step()
        num_batches = len(self.local_data['batches'])

        for batch_idx, batch in enumerate(self.local_data['batches']):
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs

            # 3) 将数据搬到对应设备
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            history = history.to(self.device)
            labels = labels.to(self.device)

            # 4) 从 self.user_neighbor_emb 提取邻居嵌入
            batch_neighbor_emb = self.user_neighbor_emb[user_ids.cpu().numpy()]
            batch_neighbor_emb = torch.tensor(batch_neighbor_emb, dtype=torch.float32).to(self.device)

            # 对 batch_neighbor_emb 做维度修正、padding 等操作
            if batch_neighbor_emb.shape[2] > NEIGHBOR_LEN:
                batch_neighbor_emb = batch_neighbor_emb[:, :, :NEIGHBOR_LEN, :]
            elif batch_neighbor_emb.shape[2] < NEIGHBOR_LEN:
                pad_size = NEIGHBOR_LEN - batch_neighbor_emb.shape[2]
                pad = torch.zeros(
                    batch_neighbor_emb.shape[0], batch_neighbor_emb.shape[1],
                    pad_size, batch_neighbor_emb.shape[3],
                    device=self.device
                )
                batch_neighbor_emb = torch.cat((batch_neighbor_emb, pad), dim=2)

            # 5) 前向 + 计算 Loss（混合精度）
            with torch.cuda.amp.autocast():
                output = self.model(user_ids, item_ids, history, batch_neighbor_emb)
                loss = torch.nn.functional.mse_loss(output, labels)
                # 若你用 Label Scale，需要这里适度除/乘
                loss = loss / accumulation_steps  # 梯度累积的等效缩放

            # 6) 若出现 NaN/Inf，跳过本 batch
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"[DEBUG] Loss is NaN/Inf in client {self.client_id}, batch {batch_idx}. Skipping this batch.")
                continue

            # 7) 反向传播（混合精度）
            self.scaler.scale(loss).backward()

            # 8) 梯度累积逻辑
            #    - 每到 accumulation_steps 次，就 step() 一次
            #    - 或者如果到了最后一个 batch (batch_idx+1 == num_batches)，也要执行一次 step()
            do_optimizer_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == num_batches)

            if do_optimizer_step:
                # 在 step 前进行梯度裁剪，防止爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # 9) 记录当前 batch 的损失（乘回 accumulation_steps 便于查看原始规模）
            batch_loss = loss.item() * accumulation_steps
            epoch_loss += batch_loss

            # 仅调试：每隔10个 batch 打印一次预测、真实值
            if batch_idx % 10 == 0:
                print(f"[Client {self.client_id}] Batch {batch_idx + 1}/{num_batches}, Loss: {batch_loss}")
                print(f"真实评分 (labels): {labels.detach().cpu().numpy()[:5]}")
                print(f"预测评分 (output): {output.detach().cpu().numpy()[:5]}")

        # 10) 训练完成后，输出平均损失
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
        else:
            avg_loss = 0.0
        print(f"[Client {self.client_id}] Epoch Loss: {avg_loss}")

        # 11) 收集并返回本地更新后的模型参数
        updated_params = [
            param.data.clone()  # param.data 而不是 param.grad
            for param in self.model.parameters()
        ]

        # 打印当前客户端最终参数均值，确认是否更新
        # for idx, p in enumerate(self.model.parameters()):
        #     print(f"Client {self.client_id} param {idx} avg: {p.data.mean().item()}")

        return updated_params
