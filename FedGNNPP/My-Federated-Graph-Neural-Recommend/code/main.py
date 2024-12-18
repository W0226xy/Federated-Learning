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
NUM_ROUNDS = 3
PATIENCE = 3  # Number of rounds to wait for improvement

path_dataset = 'training_test_dataset.mat'  # Specify dataset file path

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
    global_model = GraphRecommendationModel(num_users=num_users + 3, num_items=num_items + 3, hidden_dim=HIDDEN).to(device)
    server = FederatedServer(global_model)

    print("[INFO] Global model and server initialized.")

    # Split data among clients
    client_data_splits = split_data_for_clients(list(zip(trainu, traini, trainlabel)), NUM_CLIENTS)
    print(f"[INFO] Data split into {NUM_CLIENTS} clients.")

    # Generate batches for each client
    user_neighbor_emb = graph_embedding_expansion(Otraining, usernei,global_model.user_embedding.weight.data.cpu().numpy())
    # 生成本地数据批次
    train_batches = [
        generate_local_batches(client_data, BATCH_SIZE, user_neighbor_emb, usernei)
        for client_data in client_data_splits
    ]

    print(f"[INFO] Training batches generated for each client.")
    # Initialize clients
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

    # Federated learning loop
    for round_num in range(NUM_ROUNDS):
        print(f"\n[Round {round_num + 1}] Starting training...")

        round_loss = 0  # Accumulate round loss
        client_gradients = []
        for client in clients:
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
        test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)  # Placeholder for neighbor_emb
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        all_preds = []
        all_labels = []
        round_loss = 0
        with torch.no_grad():
            for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
                user_ids = user_ids.long().to(device)  # Ensure tensor type is consistent and move to device
                item_ids = item_ids.long().to(device)  # Ensure tensor type is consistent and move to device
                history = history.long().to(device)  # Ensure tensor type is consistent and move to device
                neighbor_emb = neighbor_emb.float().to(device)  # Ensure tensor type is consistent and move to device
                labels = labels.to(device)  # Move labels to device

                # Debugging devices
                #print(f"[DEBUG] user_ids device: {user_ids.device}, item_ids device: {item_ids.device}")
                #print(f"[DEBUG] history device: {history.device}, neighbor_emb device: {neighbor_emb.device}")
                #print(f"[DEBUG] labels device: {labels.device}, global_model device: {next(global_model.parameters()).device}")

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
    test_dataset = CustomDataset(testu, testi, testlabel, usernei, usernei)  # Placeholder for neighbor_emb
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
            user_ids = user_ids.long().to(device)  # Ensure tensor type is consistent and move to device
            item_ids = item_ids.long().to(device)  # Ensure tensor type is consistent and move to device
            history = history.long().to(device)  # Ensure tensor type is consistent and move to device
            neighbor_emb = neighbor_emb.float().to(device)  # Ensure tensor type is consistent and move to device
            labels = labels.to(device)  # Move labels to device

            # Debugging devices
            # print(f"[DEBUG] user_ids device: {user_ids.device}, item_ids device: {item_ids.device}")
            # print(f"[DEBUG] history device: {history.device}, neighbor_emb device: {neighbor_emb.device}")
            # print(f"[DEBUG] labels device: {labels.device}, global_model device: {next(global_model.parameters()).device}")

            output = global_model(user_ids, item_ids, history, neighbor_emb)
            all_preds.append(output)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    rmse = np.sqrt(np.mean(np.square(all_preds - all_labels / LABEL_SCALE))) * LABEL_SCALE
    print('Final evaluation phase:rmse:', rmse)
