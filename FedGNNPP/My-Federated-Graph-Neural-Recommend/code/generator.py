# generator.py

import numpy as np
from torch.utils.data import DataLoader
from model import CustomDataset
from const import HIS_LEN, NEIGHBOR_LEN, HIDDEN
import logging
from typing import List, Tuple, Dict, Generator

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def split_data_for_clients(data: List[Tuple[int, int, float]], num_clients: int) -> List[List[Tuple[int, int, float]]]:
    """
    将全局数据分割成多个客户端数据分片，每个客户端仅包含一个用户的数据。

    :param data: 全局数据列表，格式为 [(user_id, item_id, label), ...]
    :param num_clients: 客户端数量（应等于唯一用户数量）
    :return: 每个客户端的本地数据列表
    """
    if num_clients <= 0:
        raise ValueError("Number of clients must be positive.")
    if not isinstance(data, list):
        raise TypeError("Data must be a list of tuples (user_id, item_id, label).")

    # 按用户ID分组
    user_to_data = {}
    for user_id, item_id, label in data:
        if user_id not in user_to_data:
            user_to_data[user_id] = []
        user_to_data[user_id].append((user_id, item_id, label))

    unique_users = sorted(user_to_data.keys())
    if num_clients != len(unique_users):
        raise ValueError(f"Number of clients ({num_clients}) must equal number of unique users ({len(unique_users)}).")

    # 分配每个客户端的数据
    client_data_splits = [user_to_data[user_id] for user_id in unique_users]
    logging.debug(f"Data split into {len(client_data_splits)} clients, each with one user.")

    return client_data_splits


def generate_local_batches(
    client_data: List[Tuple[int, int, float]],
    batch_size: int,
    user_neighbor_emb: np.ndarray,
    history: np.ndarray
) -> DataLoader:
    """
    为每个客户端生成本地数据的 DataLoader。

    :param client_data: 客户端数据列表，格式为 [(user_id, item_id, label), ...]
    :param batch_size: 每个批次的大小
    :param user_neighbor_emb: 用户邻居嵌入，形状为 (num_users, HIS_LEN, NEIGHBOR_LEN, HIDDEN)
    :param history: 用户历史交互数据，形状为 (num_users, HIS_LEN, NEIGHBOR_LEN, HIDDEN)
    :return: PyTorch DataLoader 对象
    """
    if not client_data:
        raise ValueError("Client data is empty.")

    try:
        user_ids, item_ids, labels = zip(*client_data)
    except ValueError:
        raise ValueError("Client data must be a list of tuples (user_id, item_id, label).")

    user_ids = np.array(user_ids, dtype='int32')
    item_ids = np.array(item_ids, dtype='int32')
    labels = np.array(labels, dtype='float32')

    # 验证数据长度与嵌入的一致性
    if len(user_ids) != len(item_ids) or len(user_ids) != len(labels):
        raise ValueError("user_ids, item_ids, and labels must have the same length.")

    # 确保用户ID在 history 和 user_neighbor_emb 中有效
    if np.any(user_ids >= history.shape[0]):
        raise IndexError("User ID exceeds history array dimensions.")
    if np.any(user_ids >= user_neighbor_emb.shape[0]):
        raise IndexError("User ID exceeds user_neighbor_emb array dimensions.")

    dataset = CustomDataset(user_ids, item_ids, labels, history=history, neighbor_emb=user_neighbor_emb)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logging.debug(f"Generated DataLoader with batch size {batch_size} for client.")

    return dataloader