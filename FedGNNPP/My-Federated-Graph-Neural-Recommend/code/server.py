#server.py
import torch
class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model  # 全局模型
        self.global_optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.01)

    def aggregate_gradients(self, client_gradients):
        """
        聚合客户端上传的梯度并更新全局模型。
        :param client_gradients: List[Dict]，每个客户端的梯度字典。
        """
        with torch.no_grad():
            for param_index, param in enumerate(self.global_model.parameters()):
                # 聚合客户端梯度
                aggregated_grad = sum(client_grad[param_index] for client_grad in client_gradients) / len(
                    client_gradients)

                # 确保梯度与模型参数在同一设备上
                param.grad = aggregated_grad.to(param.device)

        self.global_optimizer.step()  # 更新全局模型参数

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