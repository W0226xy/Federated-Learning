import numpy as np
from torch.utils.data import DataLoader
from model import CustomDataset
from const import HIS_LEN, NEIGHBOR_LEN, HIDDEN


def generate_batch_data_random(batch_size, train_user_index, trainu, traini, history, trainlabel, user_neighbor_emb):
    idx = np.array(list(train_user_index.keys()))  # Randomize user indices
    np.random.shuffle(idx)  # Shuffle user indices to ensure different order for each batch
    batches = [idx[range(batch_size * i, min(len(idx), batch_size * (i + 1)))] for i in
               range(len(idx) // batch_size + 1) if len(range(batch_size * i, min(len(idx), batch_size * (i + 1))))]
    # Divide user indices into batches according to batch size
    while (True):  # Infinite loop to keep generating data
        for i in batches:
            idxs = [train_user_index[u] for u in i]
            uid = np.array([])
            iid = np.array([])
            uneiemb = user_neighbor_emb[:0]
            y = np.array([])
            for idss in idxs:
                uid = np.concatenate([uid, trainu[idss]])  # Concatenate user IDs
                iid = np.concatenate([iid, traini[idss]])  # Concatenate item IDs
                y = np.concatenate([y, trainlabel[idss]])  # Concatenate labels
                uneiemb = np.concatenate([uneiemb, user_neighbor_emb[trainu[idss]]],
                                         axis=0)  # Concatenate neighbor embeddings
            uid = np.array(uid, dtype='int32')
            iid = np.array(iid, dtype='int32')
            ui = history[uid]
            uid = np.expand_dims(uid, axis=1)  # Expand dimensions for model input
            iid = np.expand_dims(iid, axis=1)

            yield ([uid, iid, ui, uneiemb], [y])  # Yield generated batch data


def generate_batch_data(batch_size, testu, testi, history, testlabel, user_neighbor_emb):
    idx = np.arange(len(testlabel))
    np.random.shuffle(idx)
    y = testlabel
    batches = [idx[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            uid = np.expand_dims(testu[i], axis=1)
            iid = np.expand_dims(testi[i], axis=1)
            ui = history[testu[i]]
            uneiemb = user_neighbor_emb[testu[i]]

            yield ([uid, iid, ui, uneiemb], [y])


def split_data_for_clients(data, num_clients):
    """
    将全局数据分割成多个客户端数据分片。

    :param data: 全局数据（如 [(trainu, traini, trainlabel), ...] 的列表形式）
    :param num_clients: 客户端数量
    :return: 每个客户端的本地数据列表
    """
    data_splits = []
    split_size = len(data) // num_clients
    for i in range(num_clients):
        if i == num_clients - 1:  # 确保最后一个客户端包含剩余数据
            data_splits.append(data[i * split_size:])
        else:
            data_splits.append(data[i * split_size:(i + 1) * split_size])
    return data_splits


def generate_local_batches(client_data, batch_size):
    user_ids, item_ids, labels = zip(*client_data)

    # Debug: Check generated user, item, and label data
    print(f"[DEBUG] Generating batches:")
    print(f"  User IDs: {user_ids[:5]}")
    print(f"  Item IDs: {item_ids[:5]}")
    print(f"  Labels: {labels[:5]}")

    # Create dummy history and neighbor_emb for debugging (replace with actual data)
    history = [np.zeros(HIS_LEN, dtype=np.int32) for _ in labels]
    neighbor_emb = [np.zeros((HIS_LEN, NEIGHBOR_LEN, HIDDEN), dtype=np.float32) for _ in labels]

    dataset = CustomDataset(user_ids, item_ids, labels, history, neighbor_emb)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Debug: Log batch structures
    for batch_index, batch in enumerate(dataloader):
        inputs, labels = batch
        print(f"[DEBUG] Batch {batch_index + 1}:")
        print(f"  Inputs shapes: {[x.shape for x in inputs]}")
        print(f"  Labels shape: {labels.shape}")

    return dataloader

