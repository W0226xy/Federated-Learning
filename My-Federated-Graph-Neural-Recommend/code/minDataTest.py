import h5py
import numpy as np
import scipy.sparse as sp  # 导入 scipy.sparse

# 设置 numpy 打印选项，确保打印所有数据
np.set_printoptions(threshold=np.inf)

# 文件路径
path_dataset = 'training_test_dataset.mat'

# 读取并提取前 50 条数据的函数
def load_first_50(path_file, name_field, num_samples=50):
    with h5py.File(path_file, 'r') as db:
        # 获取数据集字段
        ds = db[name_field]

        # 检查数据集类型，如果是稀疏矩阵需要特殊处理
        try:
            if 'ir' in ds.keys():
                # 提取稀疏矩阵数据
                data = np.asarray(ds['data'])[:num_samples]
                ir = np.asarray(ds['ir'])[:num_samples]
                jc = np.asarray(ds['jc'])[:num_samples]
                out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
            else:
                out = np.asarray(ds).astype(np.float32).T
        except AttributeError:
            # 如果是密集矩阵，直接提取并转置（适应 Python 和 MATLAB 的存储格式差异）
            out = np.asarray(ds).astype(np.float32).T

        # 截取前 num_samples 条
        out = out[:num_samples]

    return out

# 保存前 50 条数据到新的 .mat 文件
def save_first_50_to_mat(filename, M, Otraining, Otest):
    with h5py.File(filename, 'w') as db:
        db.create_dataset('M', data=M)
        db.create_dataset('Otraining', data=Otraining)
        db.create_dataset('Otest', data=Otest)

if __name__ == "__main__":
    # 提取前 50 条交互数据用于调试
    try:
        M_first_50 = load_first_50(path_dataset, 'M', num_samples=50)
        Otraining_first_50 = load_first_50(path_dataset, 'Otraining', num_samples=50)
        Otest_first_50 = load_first_50(path_dataset, 'Otest', num_samples=50)

        print("前 50 条交互数据 (M):", M_first_50)
        print("前 50 条训练数据 (Otraining):", Otraining_first_50)
        print("前 50 条测试数据 (Otest):", Otest_first_50)

        # 保存前 50 条数据到新的 .mat 文件
        save_first_50_to_mat('training_test_dataset_50.mat', M_first_50, Otraining_first_50, Otest_first_50)
        print("前 50 条数据已保存到 'training_test_dataset_50.mat'")
    except FileNotFoundError:
        print("错误: 找不到文件 'training_test_dataset.mat'，请确认路径是否正确。")
