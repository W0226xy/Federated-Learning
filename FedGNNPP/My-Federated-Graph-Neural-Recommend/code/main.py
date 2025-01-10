# main.py

from utils import *
from encrypt import *
from model import *
from preprocess import *
from expansion import *
from generator import *
from const import *
from server import FederatedServer
from client import FederatedClient

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

# 请根据你的实际情况，修改为正确的文件路径
path_dataset = 'training_test_dataset.mat'
#D:\学习项目汇总\实验数据集\Flixster\\training_test_dataset_10_NNs.mat
def select_clients(available_clients, num_selected, NUM_CLIENTS):
    """
    从 available_clients 中选择 num_selected 个客户端。
    如果 available_clients 中不足 num_selected 个，则选择所有剩余的，并重置 available_clients。
    """
    if len(available_clients) < num_selected:
        selected = available_clients.copy()
        # 重置 available_clients
        available_clients = list(range(NUM_CLIENTS))
        # 从剩余需要选择的客户端中随机选择
        remaining = num_selected - len(selected)
        selected += random.sample(available_clients, remaining)
        # 移除已选择的客户端
        for client_id in selected[len(selected) - remaining:]:
            available_clients.remove(client_id)
    else:
        selected = random.sample(available_clients, num_selected)
        for client_id in selected:
            available_clients.remove(client_id)
    return selected, available_clients

if __name__ == "__main__":
    # 1. 固定随机种子（可选）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 2. 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 读取数据
    M = load_matlab_file(path_dataset, 'M')            # User-item interaction matrix
    Otraining = load_matlab_file(path_dataset, 'Otraining')  # Training interaction data matrix
    Otest = load_matlab_file(path_dataset, 'Otest')          # Testing interaction data matrix
    print('There are %i interactions logs.' % np.sum(np.array(np.array(M, dtype='bool'), dtype='int32')))

    # 4. 预处理
    usernei = generate_history(Otraining)  # 生成用户历史交互
    print(f"[DEBUG] Generated history (usernei): {usernei[:5]}")

    trainu, traini, trainlabel, train_user_index = generate_training_data(Otraining, M)
    testu, testi, testlabel = generate_test_data(Otest, M)

    unique_train_users = len(train_user_index)
    print(f"[DEBUG] Number of unique training users: {unique_train_users}")
    print("[INFO] Data preprocessed successfully.")
    print(f"[DEBUG] Training data counts - trainu: {len(trainu)}, traini: {len(traini)}, trainlabel: {len(trainlabel)}")

    # 5. 初始化全局模型 & 服务器
    num_users, num_items = Otraining.shape[0], Otraining.shape[1]
    NUM_CLIENTS = unique_train_users  # 在你的场景中，可能是 1132
    NUM_ROUNDS = 10   # 迭代轮数示例
    PATIENCE = 10    # 提前终止轮数
    SELECTED_CLIENTS_PER_ROUND = 256

    # 创建全局模型
    global_model = GraphRecommendationModel(
        num_users=num_users + 3,
        num_items=num_items + 3,
        hidden_dim=HIDDEN
    ).to(device)

    # 创建服务器
    server = FederatedServer(global_model, device=device)
    print("[INFO] Global model and server initialized.")

    # 6. 划分数据到各客户端
    data = list(zip(trainu, traini, trainlabel))
    client_data_splits = split_data_for_clients(data, NUM_CLIENTS)
    print(f"[INFO] Data split into {NUM_CLIENTS} clients.")

    # 7. 生成用户邻居嵌入
    user_neighbor_emb = graph_embedding_expansion(
        Otraining,
        usernei,
        global_model.user_embedding.weight.data.cpu().numpy()
    )
    print(f"Shape of user_neighbor_emb: {user_neighbor_emb.shape}")

    # 8. 为每个客户端生成训练批次
    train_batches = [
        generate_local_batches(client_data, BATCH_SIZE, user_neighbor_emb, usernei)
        for client_data in client_data_splits
    ]
    print(f"[INFO] Training batches generated for each client.")

    # 9. 初始化客户端（上传“模型参数” 而非“梯度”）
    #    这里注意到 train() 函数里，最后返回的是“更新后参数” (List[Tensor])。
    clients = []
    for i in range(NUM_CLIENTS):
        client_obj = FederatedClient(
            client_id=i,
            local_data={'batches': train_batches[i]},
            model=GraphRecommendationModel(num_users=num_users + 3,
                                           num_items=num_items + 3,
                                           hidden_dim=HIDDEN).to(device),
            device=device,
            user_neighbor_emb=user_neighbor_emb
        )
        clients.append(client_obj)

    print(f"[INFO] {NUM_CLIENTS} clients initialized successfully.")

    # Early stopping 相关参数
    best_loss = float('inf')
    early_stop_counter = 0

    # 客户端选择列表
    available_clients = list(range(NUM_CLIENTS))

    # 10. 联邦训练循环
    for round_num in range(NUM_ROUNDS):
        print(f"\n[Round {round_num + 1}] Starting training...")

        # 10.1 广播当前全局模型参数
        global_params = server.broadcast_model_params()

        # 10.2 随机选择部分客户端
        selected_clients_ids, available_clients = select_clients(
            available_clients, SELECTED_CLIENTS_PER_ROUND, NUM_CLIENTS
        )
        print(f"[INFO] Selected clients for this round: {selected_clients_ids}")

        # 10.3 在各客户端进行本地训练并返回“更新后模型参数”
        all_client_params = []
        for client_id in selected_clients_ids:
            client = clients[client_id]
            updated_params = client.train(
                global_model_params=global_params,
                Otraining=Otraining,
                usernei=usernei,
                global_embedding=global_model.user_embedding.weight.data.cpu().numpy()
            )
            all_client_params.append(updated_params)

        # 10.4 服务器端进行聚合，将全局模型更新
        server.aggregate_parameters(all_client_params)

        # 10.5 测试/评估全局模型
        server.global_model.eval()
        test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        round_loss = 0
        with torch.no_grad():
            for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
                user_ids = user_ids.long().to(device)
                item_ids = item_ids.long().to(device)
                history = history.long().to(device)
                neighbor_emb = neighbor_emb.float().to(device)
                labels = labels.to(device)

                output = server.global_model(user_ids, item_ids, history, neighbor_emb)
                loss = torch.nn.functional.mse_loss(output, labels)
                round_loss += loss.item()

        round_loss /= len(test_loader)
        print(f"[Round {round_num + 1}] Average Loss: {round_loss}")

        # 10.6 Early Stopping 逻辑
        if round_loss < best_loss:
            best_loss = round_loss
            early_stop_counter = 0
            print(f"[INFO] New best loss: {best_loss}")
        else:
            early_stop_counter += 1
            print(f"[INFO] No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print(f"[INFO] Early stopping triggered after {round_num + 1} rounds.")
            break

    # 11. 最终评估
    print("\n[Training Completed] Final evaluation...")
    server.global_model.eval()
    test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    all_users = []
    all_items = []

    with torch.no_grad():
        for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
            user_ids = user_ids.long().to(device)
            item_ids = item_ids.long().to(device)
            history = history.long().to(device)
            neighbor_emb = neighbor_emb.float().to(device)
            labels = labels.to(device)

            output = server.global_model(user_ids, item_ids, history, neighbor_emb)

            all_preds.append(output)
            all_labels.append(labels)
            all_users.append(user_ids.cpu().numpy())
            all_items.append(item_ids.cpu().numpy())

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_users = np.concatenate(all_users)
    all_items = np.concatenate(all_items)

    # 计算 RMSE
    rmse = np.sqrt(np.mean(np.square(all_preds - all_labels / LABEL_SCALE))) * LABEL_SCALE
    # 计算 MSE
    mse = np.mean(np.square(all_preds - all_labels / LABEL_SCALE)) * (LABEL_SCALE ** 2)

    print('Final evaluation phase: RMSE:', rmse)
    print('Final evaluation phase: MSE:', mse)

    print("\n[INFO] Sample of user-item predictions vs actuals:")
    for i in range(min(10, len(all_users))):
        user = all_users[i]
        item = all_items[i]
        actual_rating = all_labels[i]
        predicted_rating = all_preds[i]

        if isinstance(actual_rating, np.ndarray):
            actual_rating = actual_rating.item()
        if isinstance(predicted_rating, np.ndarray):
            predicted_rating = predicted_rating.item()

        print(f"User {user}, Item {item} => Actual Rating: {actual_rating:.4f}, Predicted Rating: {predicted_rating:.4f}")
