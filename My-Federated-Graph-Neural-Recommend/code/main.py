from utils import *
from encrypt import *
from model import *
from preprocess import *
from expansion import *
from generator import *
from const import *
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set visible GPU device to 1 to ensure code runs on the specified GPU
path_dataset = 'training_test_dataset_50.mat'  # Specify dataset file path

# main.py
if __name__ == "__main__":

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

    # create PyTorch dataset and dataloader
    train_dataset = CustomDataset(trainu, traini, trainlabel, usernei,
                                  usernei)  # Using usernei as a placeholder for neighbor_emb
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialize model
    model = GraphRecommendationModel(num_users=Otraining.shape[0] + 3, num_items=Otraining.shape[1] + 3,
                                     hidden_dim=HIDDEN)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # train
    model.train()
    noise_scale = 0.01  # Fixed noise scale for differential privacy
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}/{EPOCH}")
        epoch_loss = 0
        alluserembs = model.user_embedding.weight.data.clone()  # Get user embeddings
        user_neighbor_emb = graph_embedding_expansion(Otraining, usernei, alluserembs.numpy(),
                                                      privacy_needed=True)  # Calculate neighbor embeddings

        train_dataset = CustomDataset(trainu, traini, trainlabel, usernei, user_neighbor_emb)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch_idx, ((user_ids, item_ids, history, neighbor_emb), labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear gradients
            user_ids = user_ids.long()  # Ensure tensor type is consistent
            item_ids = item_ids.long()  # Ensure tensor type is consistent
            history = history.long()  # Ensure tensor type is consistent
            neighbor_emb = neighbor_emb.float()  # Ensure tensor type is consistent
            output = model(user_ids, item_ids, history, neighbor_emb)  # Forward propagation
            loss = criterion(output, labels)  # Calculate loss
            loss.backward()  # Backward propagation

            # Add differential privacy noise
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, noise_scale, size=param.size())
                        param.grad += noise  # Add pseudo-interaction noise

                        # Add local differential privacy (LDP) noise
                        ldp_noise = torch.tensor(
                            np.random.laplace(0, LR * 2 * CLIP / np.sqrt(BATCH_SIZE) / EPS, size=param.shape),
                            dtype=torch.float32)
                        param.grad += ldp_noise

            optimizer.step()  # Update weights
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss / len(train_loader):.4f}\n")

    # test (placeholder, modify as needed)
    model.eval()
    test_dataset = CustomDataset(testu, testi, testlabel, usernei,
                                 usernei)  # Using usernei as a placeholder for neighbor_emb
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (user_ids, item_ids, history, neighbor_emb), labels in test_loader:
            user_ids = user_ids.long()  # Ensure tensor type is consistent
            item_ids = item_ids.long()  # Ensure tensor type is consistent
            history = history.long()  # Ensure tensor type is consistent
            neighbor_emb = neighbor_emb.float()  # Ensure tensor type is consistent
            output = model(user_ids, item_ids, history, neighbor_emb)
            all_preds.append(output)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    rmse = np.sqrt(np.mean(np.square(all_preds - all_labels / LABEL_SCALE))) * LABEL_SCALE
    print('rmse:', rmse)
