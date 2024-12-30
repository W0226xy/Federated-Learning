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

from model import CustomDataset

path_dataset = 'training_test_dataset_50.mat'  # Specify dataset file path


def select_clients(available_clients, num_selected, NUM_CLIENTS):
    """
    从 available_clients 中选择 num_selected 个客户端。
    如果 available_clients 中不足 num_selected 个，则选择所有剩余的，并重置 available_clients。

    :param available_clients: List[int], 尚未被选中的客户端ID列表
    :param num_selected: int, 每轮选择的客户端数量
    :param NUM_CLIENTS: int, 客户端总数量
    :return: List[int], 选择的客户端ID列表, 更新后的 available_clients
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
    # Set random seeds for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    M = load_matlab_file(path_dataset, 'M')  # User-item interaction matrix
    Otraining = load_matlab_file(path_dataset, 'Otraining')  # Training interaction data matrix
    Otest = load_matlab_file(path_dataset, 'Otest')  # Testing interaction data matrix
    print('There are %i interactions logs.' % np.sum(np.array(np.array(M, dtype='bool'), dtype='int32')))

    # Preprocess data
    usernei = generate_history(Otraining)  # Generate user interaction history
    print(f"[DEBUG] Generated history (usernei): {usernei[:5]}")  # Print first 5 users' history

    trainu, traini, trainlabel, train_user_index = generate_training_data(Otraining, M)  # Generate training data
    testu, testi, testlabel = generate_test_data(Otest, M)  # Generate test data

    unique_train_users = len(train_user_index)
    print(f"[DEBUG] Number of unique training users: {unique_train_users}")  # 应输出 1132

    print("[INFO] Data preprocessed successfully.")
    print(f"[DEBUG] Training data counts - trainu: {len(trainu)}, traini: {len(traini)}, trainlabel: {len(trainlabel)}")

    # Initialize global model and server
    num_users, num_items = Otraining.shape[0], Otraining.shape[1]
    NUM_CLIENTS = unique_train_users  # 设置为 1132
    NUM_ROUNDS = 3  # 例如，设为3轮
    PATIENCE = 10  # Number of rounds to wait for improvement
    SELECTED_CLIENTS_PER_ROUND = 64  # 每轮选择的客户端数量

    global_model = GraphRecommendationModel(num_users=num_users + 3, num_items=num_items + 3, hidden_dim=HIDDEN).to(device)
    server = FederatedServer(global_model)

    print("[INFO] Global model and server initialized.")

    # Split data among clients
    data = list(zip(trainu, traini, trainlabel))
    client_data_splits = split_data_for_clients(data, NUM_CLIENTS)  # 每个客户端仅包含一个用户的数据
    print(f"[INFO] Data split into {NUM_CLIENTS} clients.")

    # Generate batches for each client
    user_neighbor_emb = graph_embedding_expansion(Otraining, usernei, global_model.user_embedding.weight.data.cpu().numpy())
    # 生成本地数据批次
    train_batches = [
        generate_local_batches(client_data, BATCH_SIZE, user_neighbor_emb, usernei)
        for client_data in client_data_splits
    ]

    # 添加调试信息以确认每个 DataLoader 的长度
    for i, dataloader in enumerate(train_batches):
        print(f"[DEBUG] Client {i} DataLoader has {len(dataloader)} batches.")

    print(f"[INFO] Training batches generated for each client.")

    # Initialize clients, each client corresponds to one user
    clients = [
        FederatedClient(
            client_id=i,
            local_data={'batches': train_batches[i]},
            model=GraphRecommendationModel(num_users=num_users + 3, num_items=num_items + 3, hidden_dim=HIDDEN).to(device),
            device=device
        )
        for i in range(NUM_CLIENTS)
    ]
    print(f"[INFO] {NUM_CLIENTS} clients initialized successfully.")

    # Early stopping parameters
    best_loss = float('inf')  # Initialize best loss to infinity
    early_stop_counter = 0  # Counter for early stopping

    # Initialize client selection tracking
    available_clients = list(range(NUM_CLIENTS))  # Initially, all clients are available for selection

    # Federated learning loop
    for round_num in range(NUM_ROUNDS):
        print(f"\n[Round {round_num + 1}] Starting training...")

        # Select clients for this round
        selected_clients_ids, available_clients = select_clients(available_clients, SELECTED_CLIENTS_PER_ROUND, NUM_CLIENTS)
        print(f"[INFO] Selected clients for this round: {selected_clients_ids}")

        round_loss = 0  # Accumulate round loss
        client_gradients = []
        for client_id in selected_clients_ids:
            client = clients[client_id]
            print(f"[INFO] Client {client.client_id} starts training.")
            client_gradient = client.train(
                global_model.state_dict(), Otraining, usernei, global_model.user_embedding.weight.data.cpu().numpy()
            )
            client_gradients.append(client_gradient)
            print(f"[INFO] Client {client.client_id} finished training.")

        # Server aggregates gradients and updates global model
        print("[INFO] Server aggregating gradients.")
        server.aggregate_gradients(client_gradients)
        print(f"[Round {round_num + 1}] Training completed. Global model updated.")

        # Evaluation phase (calculate round loss)
        global_model.eval()
        test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)  # 注意：需要正确的 neighbor_emb
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        all_preds = []
        all_labels = []
        round_loss = 0
        with torch.no_grad():
            for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
                user_ids = user_ids.long().to(device)  # Ensure tensor type is consistent and move to device
                item_ids = item_ids.long().to(device)
                history = history.long().to(device)
                neighbor_emb = neighbor_emb.float().to(device)
                labels = labels.to(device)

                output = global_model(user_ids, item_ids, history, neighbor_emb)
                loss = torch.nn.functional.mse_loss(output, labels)  # Compute loss for evaluation
                round_loss += loss.item()

        round_loss /= len(test_loader)  # Average loss over all test batches
        print(f"[Round {round_num + 1}] Average Loss: {round_loss}")

        # Early stopping logic
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

    print("\n[Training Completed] Final evaluation...")

    # Final evaluation phase
    global_model.eval()
    test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)  # 注意：需要正确的 neighbor_emb
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

            # Forward pass
            output = global_model(user_ids, item_ids, history, neighbor_emb)

            # Store predictions and labels for evaluation
            all_preds.append(output)
            all_labels.append(labels)

            # Collect user-item pairs for displaying
            all_users.append(user_ids.cpu().numpy())
            all_items.append(item_ids.cpu().numpy())

    # Concatenate all predictions, labels, users, and items
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_users = np.concatenate(all_users)
    all_items = np.concatenate(all_items)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(all_preds - all_labels / LABEL_SCALE))) * LABEL_SCALE

    # Calculate MSE
    mse = np.mean(np.square(all_preds - all_labels / LABEL_SCALE)) * LABEL_SCALE ** 2  # Square RMSE to get MSE

    # Print results
    print('Final evaluation phase: RMSE:', rmse)
    print('Final evaluation phase: MSE:', mse)

    # Optionally, display some examples (first 10 for example)
    print("\n[INFO] Sample of user-item predictions vs actuals:")
    for i in range(min(10, len(all_users))):  # Display the first 10 samples from the test set
        user = all_users[i]  # 获取用户 ID
        item = all_items[i]  # 获取物品 ID
        actual_rating = all_labels[i]  # 获取实际评分
        predicted_rating = all_preds[i]  # 获取预测评分

        # 如果评分是 ndarray，则取第一个元素（标量值）
        if isinstance(actual_rating, np.ndarray):
            actual_rating = actual_rating.item()

        if isinstance(predicted_rating, np.ndarray):
            predicted_rating = predicted_rating.item()

        # 输出用户、物品、实际评分和预测评分
        print(
            f"User {user}, Item {item} => Actual Rating: {actual_rating:.4f}, Predicted Rating: {predicted_rating:.4f}")



