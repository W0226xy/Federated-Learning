# server.py

import torch


class FederatedServer:
    def __init__(self, global_model, device=None):
        """
        :param global_model: 全局模型 (nn.Module)，在 main.py 或其他地方初始化后传入
        :param device: 训练设备，如 "cpu" 或 "cuda"
        """
        self.global_model = global_model
        if device is None:
            # 若未显式指定 device，就根据 global_model 的第一个参数来自动推断
            self.device = next(self.global_model.parameters()).device
        else:
            self.device = device
        self.global_model.to(self.device)

        print("[SERVER] FederatedServer initialized on device:", self.device)

    def broadcast_model_params(self):
        """
        返回当前服务器的全局模型参数 (state_dict)，
        用于分发给各客户端，令它们加载并开始本地训练。
        """
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}

    def aggregate_parameters(self, all_client_params):
        """
        将客户端上传的“模型参数”进行聚合 (例如简单平均)，并更新到服务器端的 global_model 中。

        :param all_client_params: List of List[Tensor]
            - 每个元素是一个客户端上传的本地模型参数列表（与 self.global_model.parameters() 对应）
            - 例如 [ [c1_param0, c1_param1, ...], [c2_param0, c2_param1, ...], ... ]
        """
        if not all_client_params:
            print("[SERVER] No client parameters to aggregate.")
            return

        num_clients = len(all_client_params)
        print(f"[SERVER] Aggregating parameters from {num_clients} clients...")

        # 取服务器端当前的全局模型参数 (需要保证顺序与客户端参数顺序对应)
        global_params = [p for p in self.global_model.parameters()]

        # 挨个层地进行聚合
        for param_idx, param in enumerate(global_params):
            # 建立一个与 param.data 形状一致的张量，用来累加
            agg_param = torch.zeros_like(param.data, device=self.device)

            # 将所有客户端的对应层参数做加和
            for client_id, client_param_list in enumerate(all_client_params):
                # client_param_list[param_idx] 是客户端的第 param_idx 个层的参数
                agg_param += client_param_list[param_idx].to(self.device)

            # 取平均
            agg_param /= float(num_clients)

            # 将聚合结果赋给全局模型对应层
            param.data.copy_(agg_param)

        print("[SERVER] Global model updated by aggregated client parameters.")

        # ========== 打印调试信息：查看全局模型参数是否变化 ==========
        # 以“均值”为例快速查看各层数值情况
        for idx, p in enumerate(self.global_model.parameters()):
            # p.data.mean() 可能是一个标量 Tensor，用 .item() 转为浮点数
            mean_val = p.data.mean().item()
            print(f"[DEBUG] After aggregation, global param {idx} mean: {mean_val:.6f}")

        print("[SERVER] Finished aggregation and parameter inspection.")
