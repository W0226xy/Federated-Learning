import torch
from const import HIS_LEN, NEIGHBOR_LEN, HIDDEN
import numpy as np
class FederatedClient:
    def __init__(self, client_id, local_data, model, device):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        print(f"[DEBUG] Client {self.client_id} initialized with local data:")
        print(f"  Number of batches: {len(self.local_data['batches'])}")
        for batch_index, batch in enumerate(self.local_data['batches']):
            print(f"  Batch {batch_index + 1}: {batch}")

    def train(self, global_model_params, user_neighbor_emb):
        self.model.load_state_dict(global_model_params)
        self.model.to(self.device)
        self.model.train()

        print(f"[DEBUG] Client {self.client_id} begins training. Number of batches: {len(self.local_data['batches'])}")

        for batch in self.local_data['batches']:
            inputs, labels = batch
            user_ids, item_ids, history, neighbor_emb = inputs

            self.optimizer.zero_grad()

            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            history = history.to(self.device)
            neighbor_emb = neighbor_emb.to(self.device)
            labels = labels.to(self.device)

            if isinstance(user_neighbor_emb, np.ndarray):
                user_neighbor_emb = torch.tensor(user_neighbor_emb, dtype=torch.float32)

            if user_neighbor_emb.shape[2] > NEIGHBOR_LEN:
                user_neighbor_emb = user_neighbor_emb[:, :, :NEIGHBOR_LEN, :]
            elif user_neighbor_emb.shape[2] < NEIGHBOR_LEN:
                pad_size = NEIGHBOR_LEN - user_neighbor_emb.shape[2]
                pad = torch.zeros(user_neighbor_emb.shape[0], user_neighbor_emb.shape[1], pad_size, user_neighbor_emb.shape[3], device=self.device)
                user_neighbor_emb = torch.cat((user_neighbor_emb, pad), dim=2)

            user_neighbor_emb = user_neighbor_emb.to(self.device)

            print(f"[DEBUG] Processing batch:")
            print(f"  User IDs shape: {user_ids.shape}, Item IDs shape: {item_ids.shape}")
            print(f"  History shape: {history.shape}, Neighbor_emb shape: {neighbor_emb.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  User Neighbor Emb shape: {user_neighbor_emb.shape}")
            print(f"  Sample User IDs: {user_ids[:5]}")
            print(f"  Sample Item IDs: {item_ids[:5]}")
            print(f"  Sample Labels: {labels[:5]}")

            output = self.model(user_ids, item_ids, history, user_neighbor_emb)
            loss = torch.nn.functional.mse_loss(output, labels)

            loss.backward()
            self.optimizer.step()

        gradients = [param.grad.clone() for param in self.model.parameters()]
        return gradients
