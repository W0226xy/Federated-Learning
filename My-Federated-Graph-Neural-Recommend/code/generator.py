import numpy as np
from encrypt import encrypt_user_data

def generate_batch_data_random(batch_size, train_user_index, trainu, traini, history, trainlabel, user_neighbor_emb, encrypt_data_flag=False):
    idx = np.array(list(train_user_index.keys()))  # Randomize user indices
    np.random.shuffle(idx)  # Shuffle user indices to ensure different order for each batch
    batches = [idx[range(batch_size * i, min(len(idx), batch_size * (i + 1)))] for i in range(len(idx) // batch_size + 1) if len(range(batch_size * i, min(len(idx), batch_size * (i + 1))))]
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
                uneiemb = np.concatenate([uneiemb, user_neighbor_emb[trainu[idss]]], axis=0)  # Concatenate neighbor embeddings
            uid = np.array(uid, dtype='int32')
            iid = np.array(iid, dtype='int32')
            ui = history[uid]
            uid = np.expand_dims(uid, axis=1)  # Expand dimensions for model input
            iid = np.expand_dims(iid, axis=1)

            # Encrypt user data if flag is set
            if encrypt_data_flag:
                uid = encrypt_user_data(uid)
                iid = encrypt_user_data(iid)
                ui = encrypt_user_data(ui)
                uneiemb = encrypt_user_data(uneiemb)

            yield ([uid, iid, ui, uneiemb], [y])  # Yield generated batch data

def generate_batch_data(batch_size, testu, testi, history, testlabel, user_neighbor_emb, encrypt_data_flag=False):
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

            # Encrypt user data if flag is set
            if encrypt_data_flag:
                uid = encrypt_user_data(uid)
                iid = encrypt_user_data(iid)
                ui = encrypt_user_data(ui)
                uneiemb = encrypt_user_data(uneiemb)

            yield ([uid, iid, ui, uneiemb], [y])
