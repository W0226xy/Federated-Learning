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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Mixed precision gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()

        print(f"[DEBUG] Client {self.client_id} initialized with local data:")
        print(f"  Number of batches: {len(self.local_data['batches'])}")
        for batch_index, batch in enumerate(self.local_data['batches']):
            print(f"  Batch {batch_index + 1}: {batch}")

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
            user_ids, item_ids, history, _ = inputs

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
