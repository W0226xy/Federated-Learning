from encrypt import *
from const import *
import random
import numpy as np
import base64
from tqdm import tqdm


def graph_embedding_expansion(Otraining, usernei, alluserembs, privacy_needed=False):
    # local encryption
    local_ciphertext = []
    for i in tqdm(usernei):  # For each user's neighbors, generate encrypted signatures
        messages = []
        for j in i:
            if j != Otraining.shape[1] + 2:
                # If privacy is needed, apply perturbation before signing
                if privacy_needed:
                    perturbed_j = perturb_items([str(j)])  # Perturb item ID
                    j = perturbed_j[0]
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)  # Store encrypted neighbor IDs

    # local id-ciphertext mapping
    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'): j for j in
                          range(Otraining.shape[1] + 3)}  # Create mapping for ciphertext to original IDs

    # assume the local_ciphertext has been sent to the third-party server

    cipher2userid = {}  # Sending data to third-party server
    for userid, i in enumerate(local_ciphertext):  # Create a reverse mapping from ciphertext to user ID
        for j in i:
            if j not in cipher2userid:
                cipher2userid[j] = [userid]
            else:
                cipher2userid[j].append(userid)

    # third-party server prepares data

    send_data = []  # Prepare neighbor information for each user to send to third-party server
    for userid, i in tqdm(enumerate(local_ciphertext)):
        neighbor_info = {}
        for j in i:
            neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
            if len(neighbor_id):
                neighbor_info[j] = neighbor_id
        send_data.append(neighbor_info)

    # third-party server distributes send_data

    # local clients expand graphs
    # Receive neighbor information from third-party server and use it to expand local user-item graph
    user_neighbor_emb = []  # Initialize list to store neighbor embeddings
    for userid, user_items in tqdm(enumerate(usernei)):
        receive_data = send_data[userid]
        decrypted_data = {local_mapping_dict[item_key]: receive_data[item_key] for item_key in receive_data}
        all_neighbor_embs = []
        for item in user_items:
            if item in decrypted_data:
                neighbor_embs = decrypted_data[item]
                random.shuffle(neighbor_embs)
                neighbor_embs = neighbor_embs[:NEIGHBOR_LEN]
                neighbor_embs += [[0.] * HIDDEN] * (NEIGHBOR_LEN - len(neighbor_embs))
            else:
                neighbor_embs = [[0.] * HIDDEN] * NEIGHBOR_LEN  # Fill with zeros if no neighbors
            all_neighbor_embs.append(neighbor_embs)
        all_neighbor_embs = all_neighbor_embs[:HIS_LEN]
        all_neighbor_embs += [[[0.] * HIDDEN] * NEIGHBOR_LEN] * (HIS_LEN - len(all_neighbor_embs))
        all_neighbor_embs = np.zeros((HIS_LEN, NEIGHBOR_LEN, HIDDEN), dtype='float32')
        for idx, emb in enumerate(all_neighbor_embs):
            all_neighbor_embs[idx, :len(emb), :] = np.array(emb, dtype='float32')
        user_neighbor_emb.append(all_neighbor_embs)

    # Pad user_neighbor_emb to ensure consistent dimensions
    max_len = max([user.shape[0] for user in user_neighbor_emb])
    padded_user_neighbor_emb = []
    for user in user_neighbor_emb:
        pad_len = max_len - user.shape[0]
        if pad_len > 0:
            padding = np.zeros((pad_len, NEIGHBOR_LEN, HIDDEN), dtype='float32')
            padded_user = np.concatenate([user, padding], axis=0)
        else:
            padded_user = user
        padded_user_neighbor_emb.append(padded_user)

    user_neighbor_emb = np.array(padded_user_neighbor_emb, dtype='float32')  # Use float32 dtype for compatibility
    return user_neighbor_emb
