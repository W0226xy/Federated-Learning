import random
import numpy as np
import base64
import collections
from tqdm import tqdm
from encrypt import *
from const import *




def graph_embedding_expansion(Otraining, usernei, alluserembs, privacy_needed=False, all_item_ids=None, user_ids=None):
    #  如果提供了user_ids，则只处理特定的用户
    if user_ids is not None:
        usernei = [usernei[u] for u in user_ids]

    # local encryption
    local_ciphertext = []
    for i in tqdm(usernei):  # 对每个用户的邻居生成加密签名
        messages = []
        for j in i:
            if j != Otraining.shape[1] + 2:
                # 如果隐私保护开启，则对物品ID进行扰动
                if privacy_needed:
                    perturbed_j = perturb_items([str(j)], all_item_ids)  # 扰动物品ID
                    j = int(perturbed_j[0])
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)  # 存储加密后的邻居ID

    # 创建加密签名与物品ID的映射
    max_neighbor_id = max([max(user) if len(user) > 0 else 0 for user in usernei])
    mapping_range = max(max_neighbor_id + 1, Otraining.shape[1] + 3)

    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'): j for j in range(mapping_range)}

    # 假设local_ciphertext已经发送到第三方服务器

    # 构建从加密签名到用户ID的反向映射
    cipher2userid = {}
    for userid, i in enumerate(local_ciphertext):
        for j in i:
            if j not in cipher2userid:
                cipher2userid[j] = [userid]
            else:
                cipher2userid[j].append(userid)

    # 第三方服务器处理邻居嵌入
    send_data = []  # 准备发送给第三方服务器的数据
    for userid, i in tqdm(enumerate(local_ciphertext)):
        neighbor_info = {}
        for j in i:
            neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
            if len(neighbor_id):
                neighbor_info[j] = neighbor_id
        send_data.append(neighbor_info)

    # 扩展本地图
    user_neighbor_emb = []
    for userid, user_items in tqdm(enumerate(usernei)):
        receive_data = send_data[userid]
        decrypted_data = {local_mapping_dict.get(item_key, None): receive_data[item_key] for item_key in receive_data if item_key in local_mapping_dict}

        all_neighbor_embs = []
        for item in user_items:
            if item in decrypted_data:
                neighbor_embs = decrypted_data[item]
                random.shuffle(neighbor_embs)
                neighbor_embs = neighbor_embs[:NEIGHBOR_LEN]
                neighbor_embs += [[0.] * HIDDEN] * (NEIGHBOR_LEN - len(neighbor_embs))
            else:
                neighbor_embs = [[0.] * HIDDEN] * NEIGHBOR_LEN
            all_neighbor_embs.append(neighbor_embs)
        while len(all_neighbor_embs) < HIS_LEN:
            all_neighbor_embs.append([[0.] * HIDDEN] * NEIGHBOR_LEN)

        all_neighbor_embs = np.array(all_neighbor_embs, dtype='float32')
        user_neighbor_emb.append(all_neighbor_embs)

    # 填充用户嵌入矩阵
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

    user_neighbor_emb = np.array(padded_user_neighbor_emb, dtype='float32')
    return user_neighbor_emb
