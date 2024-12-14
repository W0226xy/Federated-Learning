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
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Define constants for federated learning
NUM_CLIENTS = 5
NUM_ROUNDS = 10

path_dataset = 'training_test_dataset_50.mat'  # Specify dataset file path

# main.py
if __name__ == "__main__":

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    M = load_matlab_file(path_dataset, 'M')  # User-item interaction matrix
    Otraining = load_matlab_file(path_dataset, 'Otraining')  # Training interaction data matrix
    Otest = load_matlab_file(path_dataset, 'Otest')  # Testing interaction data matrix
    print('There are %i interactions logs.' % np.sum(np.array(np.array(M, dtype='bool'), dtype='int32')))

    # preprocess data
    usernei = generate_history(Otraining)  # Generate user interaction history
    print(f"[DEBUG] Generated history (usernei): {usernei[:5]}")  # Print first 5 users' history

    trainu, traini, trainlabel, train_user_index = generate_training_data(Otraining, M)  # Generate training data
    testu, testi, testlabel = generate_test_data(Otest, M)  # Generate test data

    print("[INFO] Data preprocessed successfully.")

    # Initialize global model and server
    num_users, num_items = Otraining.shape[0], Otraining.shape[1]
    global_model = GraphRecommendationModel(num_users=num_users + 3, num_items=num_items + 3, hidden_dim=HIDDEN)
    server = FederatedServer(global_model)

    print("[INFO] Global model and server initialized.")

    # Split data among clients
    client_data_splits = split_data_for_clients(list(zip(trainu, traini, trainlabel)), NUM_CLIENTS)
    print(f"[INFO] Data split into {NUM_CLIENTS} clients.")

    # Generate batches for each client
    train_batches = [
        generate_local_batches(client_data, BATCH_SIZE)
        for client_data in client_data_splits
    ]
    print(f"[INFO] Training batches generated for each client.")

    # Initialize clients
    clients = [
        FederatedClient(
            client_id=i,
            local_data={'batches': train_batches[i]},
            model=GraphRecommendationModel(num_users=num_users + 3, num_items=num_items + 3, hidden_dim=HIDDEN),
            device=device
        )
        for i in range(NUM_CLIENTS)
    ]
    print(f"[INFO] {NUM_CLIENTS} clients initialized successfully.")

    # Federated learning loop
    for round_num in range(NUM_ROUNDS):
        print(f"\n[Round {round_num + 1}] Starting training...")

        # Distribute extended neighbor embeddings per client
        user_neighbor_embs = []
        for client in clients:
            try:
                all_item_ids = list(range(Otraining.shape[1]))  # 全局物品ID列表
                user_neighbor_emb = graph_embedding_expansion(
                    Otraining, usernei, global_model.user_embedding.weight.data.cpu().numpy(), privacy_needed=True,
                    all_item_ids=all_item_ids
                )
                user_neighbor_embs.append(user_neighbor_emb)
                print(f"[DEBUG] Generated neighbor_emb shape for client {client.client_id}: {user_neighbor_emb.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to generate neighbor_emb for client {client.client_id}: {str(e)}")

        client_gradients = []
        for client, user_neighbor_emb in zip(clients, user_neighbor_embs):
            print(f"[INFO] Client {client.client_id} starts training.")
            client_gradient = client.train(server.distribute_model(), user_neighbor_emb)
            client_gradients.append(client_gradient)
            print(f"[INFO] Client {client.client_id} finished training.")

        # Server aggregates gradients and updates global model
        print("[INFO] Server aggregating gradients.")
        server.aggregate_gradients(client_gradients)
        print(f"[Round {round_num + 1}] Training completed. Global model updated.")

    print("\n[Training Completed] Evaluating global model...")

    # Evaluation phase (placeholder, modify as needed)
    global_model.eval()
    test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)  # Placeholder for neighbor_emb
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
            print(f"[DEBUG] Test Neighbor_emb shape: {neighbor_emb.shape}")
            print(f"[DEBUG] Test Neighbor_emb sample (first user, first item): {neighbor_emb[0][0][:5]}")

            user_ids = user_ids.long().to(device)  # Ensure tensor type is consistent and move to device
            item_ids = item_ids.long().to(device)  # Ensure tensor type is consistent and move to device
            history = history.long().to(device)  # Ensure tensor type is consistent and move to device
            neighbor_emb = neighbor_emb.float().to(device)  # Ensure tensor type is consistent and move to device
            labels = labels.to(device)  # Move labels to device
            output = global_model(user_ids, item_ids, history, neighbor_emb)
            all_preds.append(output)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    rmse = np.sqrt(np.mean(np.square(all_preds - all_labels / LABEL_SCALE))) * LABEL_SCALE
    print('rmse:', rmse)
