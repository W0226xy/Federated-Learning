import h5py
import numpy as np
import scipy as sp
from encrypt import encrypt_user_data, decrypt_user_data

def load_matlab_file(path_file, name_field, encrypt_data_flag=False):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        else:
            out = np.asarray(ds).astype(np.float32).T
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()

    # Encrypt data if flag is set
    if encrypt_data_flag:
        out = encrypt_user_data(out)

    return out

def save_matlab_file(path_file, name_field, data, decrypt_data_flag=False):
    if decrypt_data_flag:
        data = decrypt_user_data(data)

    with h5py.File(path_file, 'w') as db:
        db.create_dataset(name_field, data=data)

def calculate_accuracy(preds, labels, threshold=0.5):
    preds_binary = (preds >= threshold).astype(int)
    labels_binary = (labels >= threshold).astype(int)
    correct = (preds_binary == labels_binary).sum()
    total = len(labels)
    accuracy = correct / total
    return accuracy

def log_client_contributions(client_data_sizes, aggregated_weights):
    """
    Log the contribution of each client based on data size and aggregation weights.

    :param client_data_sizes: A dictionary where keys are client IDs and values are the number of data points each client has.
    :param aggregated_weights: A dictionary where keys are client IDs and values are the weights used in aggregation.
    """
    print("Client Contributions During Aggregation:")
    for client_id in client_data_sizes:
        data_size = client_data_sizes[client_id]
        weight = aggregated_weights.get(client_id, 0)
        print(f"Client {client_id}: Data Size = {data_size}, Aggregation Weight = {weight}")

def calculate_aggregation_weights(client_data_sizes):
    """
    Calculate weights for each client based on the amount of data they contribute.

    :param client_data_sizes: A dictionary where keys are client IDs and values are the number of data points each client has.
    :return: A dictionary where keys are client IDs and values are the calculated aggregation weights.
    """
    total_data_points = sum(client_data_sizes.values())
    weights = {client_id: data_size / total_data_points for client_id, data_size in client_data_sizes.items()}
    return weights
