#server.py
import torch
class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model  # 全局模型
        self.global_optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.01)

    def aggregate_gradients(self, client_gradients, selected_client_ids):
        """
        聚合客户端上传的梯度并更新全局模型。
        :param client_gradients: List[Dict]，每个客户端的梯度字典。
        :param selected_client_ids: List[int]，当前轮次被选中的客户端的ID。
        """
        with torch.no_grad():
            # 确保每个 selected_client_ids 中的 ID 都在 client_gradients 范围内
            for client_id in selected_client_ids:
                if client_id >= len(client_gradients):
                    print(
                        f"Warning: Selected client ID {client_id} is out of range. Total clients: {len(client_gradients)}")
                    continue  # Skip this client if ID is out of range

            # 只选取当前轮次的客户端的梯度
            selected_gradients = [client_gradients[i] for i in selected_client_ids if i < len(client_gradients)]

            # 遍历全局模型的每个参数进行梯度聚合
            for param_index, param in enumerate(self.global_model.parameters()):
                print(f"Aggregating gradients for parameter {param_index}...")

                # 逐个客户端的梯度进行聚合
                aggregated_grad = None
                for i, client_grad in enumerate(selected_gradients):
                    grad = client_grad[param_index]
                    client_id = selected_client_ids[i]  # 当前客户端ID

                    print(f"Client {client_id} - Parameter {param_index} Gradient: {grad}")

                    # 检查每个客户端的梯度是否是NaN或Inf
                    if torch.any(torch.isnan(grad)) or torch.any(torch.isinf(grad)):
                        print(f"NaN or Inf gradient detected in client {client_id}, parameter {param_index}")

                    # 聚合梯度
                    if aggregated_grad is None:
                        aggregated_grad = grad
                    else:
                        aggregated_grad += grad

                # 计算平均梯度
                if aggregated_grad is not None:
                    aggregated_grad /= len(selected_gradients)

                    # 确保梯度与模型参数在同一设备上
                    param.grad = aggregated_grad.to(param.device)

                    # 打印全局模型的梯度和参数
                    if param.grad is not None:
                        print(f"Global Model - Parameter {param_index} Value: {param.data}")
                        print(f"Global Model - Parameter {param_index} Gradient: {param.grad}")

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)

        # 更新全局模型
        self.global_optimizer.step()

    def distribute_model(self):
        """
        分发全局模型参数到客户端。
        :return: 全局模型的状态字典。
        """
        return self.global_model.state_dict()

    def load_client_gradients(self, gradients):
        """
        从客户端接收梯度，用于后续聚合。
        :param gradients: List[Dict]，客户端上传的梯度。
        """
        return gradients