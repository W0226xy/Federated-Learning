LABEL_SCALE = 100
HIDDEN = 16  # Embedding dimension
DROP = 0.2
BATCH_SIZE = 32  # Batch size for training
HIS_LEN = 50  # Length of historical interactions
PSEUDO = 1000
NEIGHBOR_LEN = 100  # Maximum number of neighbor nodes considered
CLIP = 0.1
LR = 0.01
EPS = 1
EPOCH = 3

# Federated Learning Parameters
AGGREGATION_ROUNDS = 10  # Number of federated aggregation rounds
CLIENTS_PER_ROUND = 5  # Number of clients selected per round
GLOBAL_LR = 0.001  # Learning rate for global model updates
