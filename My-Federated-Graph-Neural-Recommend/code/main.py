# main.py

from utils import *
# from encrypt import *
from model import *
from preprocess import *
from expansion import *
# from generator import *
from const import *
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from server import Server
from client import Client  # Import Client class

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
    trainu, traini, trainlabel, train_user_index = generate_training_data(Otraining, M)  # Generate training data
    testu, testi, testlabel = generate_test_data(Otest, M)  # Generate test data

    # generate public & private keys
    generate_key()  # Generate public and private keys

    # split training data among clients
    num_clients = 5  # Define the number of clients
    trainu_splits = split_data_among_clients(trainu, num_clients)
    traini_splits = split_data_among_clients(traini, num_clients)
    trainlabel_splits = split_data_among_clients(trainlabel, num_clients)

    # Calculate user neighbor embeddings (privacy preserved)
    alluserembs = GraphRecommendationModel(num_users=Otraining.shape[0] + 3, num_items=Otraining.shape[1] + 3, hidden_dim=HIDDEN).user_embedding.weight.data.clone().cpu()
    user_neighbor_emb = graph_embedding_expansion(Otraining, usernei, alluserembs.numpy(), privacy_needed=True)

    # create clients
    clients = []
    for client_id in range(num_clients):
        client_dataset = CustomDataset(
            trainu_splits[client_id],
            traini_splits[client_id],
            trainlabel_splits[client_id],
            usernei,
            user_neighbor_emb  # Pass the expanded neighbor embeddings to the client
        )
        model = GraphRecommendationModel(
            num_users=Otraining.shape[0] + 3,
            num_items=Otraining.shape[1] + 3,
            hidden_dim=HIDDEN
        ).to(device)
        clients.append(Client(client_id, client_dataset, model, device))

    # initialize server with clients
    server = Server(
        client_list=clients,
        model=[clients[0].model, GraphRecommendationModel(num_users=Otraining.shape[0] + 3, num_items=Otraining.shape[1] + 3, hidden_dim=HIDDEN).to(device)],
        user_features=usernei,
        item_features=usernei,
        args=type('', (), {'device': device, 'lr': LR, 'weight_decay': 0.01})()
    )

    # train
    model_user = server.model_user
    model_item = server.model_item
    model_user.train()
    model_item.train()
    optimizer = torch.optim.SGD(list(model_user.parameters()) + list(model_item.parameters()), lr=LR)  # Add optimizer
    criterion = nn.MSELoss()  # Define the criterion for loss calculation
    noise_scale = 0.01  # Fixed noise scale for differential privacy
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}/{EPOCH}")
        epoch_loss = 0

        # Update each client's neighbor embeddings before training
        for client in clients:
            client.user_neighbor_emb = user_neighbor_emb

        for client in clients:
            client.model.train()
            client_data_loader = DataLoader(client.data, batch_size=BATCH_SIZE, shuffle=True)
            for batch_idx, ((user_ids, item_ids, history, neighbor_emb), labels) in enumerate(client_data_loader):
                optimizer.zero_grad()  # Clear gradients
                user_ids = user_ids.long().to(device)  # Ensure tensor type is consistent and move to device
                item_ids = item_ids.long().to(device)  # Ensure tensor type is consistent and move to device
                history = history.long().to(device)  # Ensure tensor type is consistent and move to device
                neighbor_emb = neighbor_emb.float().to(device)  # Ensure tensor type is consistent and move to device
                labels = labels.to(device)  # Move labels to device
                output_user = client.model(user_ids, item_ids, history, neighbor_emb)  # Forward propagation for user model
                loss_user = torch.sqrt(criterion(output_user, labels))  # Calculate RMSE loss for user model
                loss = loss_user
                loss.backward()  # Backward propagation

                #打印每个参数的梯度
                for name, param in client.model.named_parameters():
                    if param.grad is None:
                        print(f"Warning: {name} has no gradient.")
                    else:
                        print(f"{name} grad: {param.grad.sum().item()}")  # 打印梯度的和

                # Add differential privacy noise
                with torch.no_grad():
                    for param in client.model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(0, noise_scale, size=param.size(), device=device)
                            param.grad += noise  # Add pseudo-interaction noise

                            # Add local differential privacy (LDP) noise
                            ldp_noise = torch.tensor(
                                np.random.laplace(0, LR * 2 * CLIP / np.sqrt(BATCH_SIZE) / EPS, size=param.shape),
                                dtype=torch.float32, device=device)
                            param.grad += ldp_noise

                optimizer.step()  # Update weights
                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Client {client.client_id}, Batch {batch_idx + 1}/{len(client_data_loader)}, RMSE Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss / len(clients):.4f}\n")

        # Aggregate parameters at the server
        server.aggregate(param_list=[{
            'model': (list(model_user.parameters()), list(model_item.parameters())),
            'item': (model_item.item_embedding.weight.grad, traini),
            'user': (model_user.user_embedding.weight.grad, trainu),
            'num_data_points': len(trainu)  # 添加客户端数据量
        }])
        # Train global GAT
        server.global_graph_training()

    # test (placeholder, modify as needed)
    model_user.eval()
    model_item.eval()
    test_dataset = CustomDataset(testu, testi, testlabel, usernei,
                                 usernei)  # Using usernei as a placeholder for neighbor_emb
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
            output_user = model_user(user_ids, item_ids, history, neighbor_emb)
            output_item = model_item(user_ids, item_ids, history, neighbor_emb)
            output = (output_user + output_item) / 2
            all_preds.append(output)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    rmse = np.sqrt(np.mean(np.square(all_preds - all_labels / LABEL_SCALE))) * LABEL_SCALE
    accuracy = calculate_accuracy(all_preds, all_labels)
    print('rmse:', rmse)
    print('accuracy:', accuracy)
