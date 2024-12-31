# client.py

import torch
from const import HIS_LEN, NEIGHBOR_LEN, HIDDEN
import numpy as np
from expansion import graph_embedding_expansion

class FederatedClient:
    def __init__(self, client_id, local_data, model, device):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Mixed precision gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()

        print(f"[DEBUG] Client {self.client_id} initialized with local data:")
        # Check the number of batches
        print(f"  Number of batches: {len(self.local_data['batches'])}")
        # Iterate over the first few batches for debugging
        for batch_index, batch in enumerate(self.local_data['batches']):
            if batch_index >= 2:  # Only print first 2 batches to avoid clutter
                break
            print(f"  Batch {batch_index + 1}:")
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs
            print(f"    user_ids: {user_ids[:5]}")
            print(f"    item_ids: {item_ids[:5]}")
            print(f"    history: {history[:5]}")
            print(f"    neighbor_emb: {neighbor_emb[:5]}")
            print(f"    labels: {labels[:5]}")

    def train(self, global_model_params, Otraining, usernei, global_embedding):
        self.model.load_state_dict(global_model_params)
        self.model.to(self.device)
        self.model.train()

        print(f"[DEBUG] Client {self.client_id} begins training. Number of batches: {len(self.local_data['batches'])}")

        accumulation_steps = 4  # Gradient accumulation steps
        epoch_loss = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.local_data['batches']):
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs

            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            history = history.to(self.device)
            labels = labels.to(self.device)

            # Dynamically generate neighbor embeddings
            batch_neighbor_emb = graph_embedding_expansion(
                Otraining, usernei, global_embedding, user_ids=user_ids.cpu().numpy()
            )
            batch_neighbor_emb = torch.tensor(batch_neighbor_emb, dtype=torch.float32).to(self.device)

            # Adjust dimensions
            if batch_neighbor_emb.shape[2] > NEIGHBOR_LEN:
                batch_neighbor_emb = batch_neighbor_emb[:, :, :NEIGHBOR_LEN, :]
            elif batch_neighbor_emb.shape[2] < NEIGHBOR_LEN:
                pad_size = NEIGHBOR_LEN - batch_neighbor_emb.shape[2]
                pad = torch.zeros(
                    batch_neighbor_emb.shape[0], batch_neighbor_emb.shape[1], pad_size, batch_neighbor_emb.shape[3],
                    device=self.device
                )
                batch_neighbor_emb = torch.cat((batch_neighbor_emb, pad), dim=2)

            # Forward pass with autocast for mixed precision
            with torch.cuda.amp.autocast():
                output = self.model(user_ids, item_ids, history, batch_neighbor_emb)
                loss = torch.nn.functional.mse_loss(output, labels)
                loss = loss / accumulation_steps  # Normalize loss for accumulation

            # Print real and predicted ratings for debugging
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"[Client {self.client_id}] Batch {batch_idx + 1}/{len(self.local_data['batches'])}")
                print(f"真实评分 (labels): {labels.cpu().numpy()[:5]}")  # Show the first 5 for brevity
                print(f"预测评分 (output): {output.cpu().detach().numpy()[:5]}")  # Show the first 5 for brevity

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Print batch loss
            print(f"[Client {self.client_id}] Batch {batch_idx + 1}/{len(self.local_data['batches'])}, Loss: {loss.item() * accumulation_steps}")
            epoch_loss += loss.item() * accumulation_steps

        print(f"[Client {self.client_id}] Epoch Loss: {epoch_loss / len(self.local_data['batches'])}")
        gradients = [
            param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in self.model.parameters()
        ]
        return gradients