#expansion.py
import random
import numpy as np
import base64
import collections
from tqdm import tqdm
from encrypt import *
from const import *


# 步骤1: 构建倒排索引，将物品ID映射到所有与其交互的用户ID
def build_item_to_users_index(Otraining):
    """
    构建倒排索引，将物品ID映射到所有与该物品有交互的用户ID。
    """
    item_to_users = collections.defaultdict(set)

    # 遍历 Otraining 矩阵
    for user_id in range(Otraining.shape[0]):
        for item_id in range(Otraining.shape[1]):
            if Otraining[user_id][item_id] != 0:  # 如果用户对物品有交互
                item_to_users[item_id].add(user_id)

    return item_to_users


# 步骤2: 使用倒排索引查找与当前用户相同交互物品的邻居
def find_nearest_neighbors(user_id, Otraining, item_to_users, top_k=10):
    """
    根据用户与物品的交互，查找与该用户有相同交互物品的其他用户。
    """
    user_items = set(np.where(Otraining[user_id] != 0)[0])  # 获取用户交互过的所有物品ID
    neighbor_candidates = collections.defaultdict(int)  # 记录每个候选邻居与当前用户的共同交互次数

    # 对于每个用户交互过的物品，找到与这些物品交互的其他用户
    for item_id in user_items:
        for neighbor_id in item_to_users[item_id]:
            if neighbor_id != user_id:  # 排除当前用户自己
                neighbor_candidates[neighbor_id] += 1  # 记录共同交互的次数

    # 按照共同交互的次数降序排序，选出前 top_k 个邻居
    nearest_neighbors = sorted(neighbor_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [neighbor_id for neighbor_id, _ in nearest_neighbors]


def graph_embedding_expansion(Otraining, usernei, alluserembs, privacy_needed=False, all_item_ids=None, user_ids=None):
    """
    Generate neighbor embeddings only for specified user_ids.
    If user_ids is None, process all users in usernei.
    """
    # Filter usernei based on user_ids if provided
    if user_ids is not None:
        usernei = [usernei[u] for u in user_ids]

    # local encryption
    local_ciphertext = []
    for i in tqdm(usernei):  # For each user's neighbors, generate encrypted signatures
        messages = []
        for j in i:
            if j != Otraining.shape[1] + 2:
                # If privacy is needed, apply perturbation before signing
                if privacy_needed:
                    perturbed_j = perturb_items([str(j)], all_item_ids)  # Perturb item ID with all_item_ids
                    j = int(perturbed_j[0])
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)  # Store encrypted neighbor IDs

    # Extend the range of local_mapping_dict to ensure complete coverage
    max_neighbor_id = max([max(user) if len(user) > 0 else 0 for user in usernei])
    mapping_range = max(max_neighbor_id + 1, Otraining.shape[1] + 3)

    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'): j for j in range(mapping_range)}

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
        decrypted_data = {local_mapping_dict.get(item_key, None): receive_data[item_key] for item_key in receive_data if
                          item_key in local_mapping_dict}

        # Debug missing keys
        missing_keys = [item_key for item_key in receive_data if item_key not in local_mapping_dict]
        if missing_keys:
            print(f"[DEBUG] Missing keys in local_mapping_dict for user {userid}: {missing_keys[:5]}")
            print(f"[DEBUG] Total missing keys: {len(missing_keys)}")

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
        while len(all_neighbor_embs) < HIS_LEN:
            all_neighbor_embs.append([[0.] * HIDDEN] * NEIGHBOR_LEN)  # Pad history to HIS_LEN

        all_neighbor_embs = np.array(all_neighbor_embs, dtype='float32')
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