import torch
print(torch.cuda.is_available())  # 如果返回 True，表示 GPU 可用
print(torch.cuda.device_count())  # 打印可用 GPU 的数量
