#const.py
LABEL_SCALE = 1#标签缩放比例
HIDDEN=64#嵌入维度
DROP=0.3#DropOut比率，防止过拟合
BATCH_SIZE=64#每批训练大小
HIS_LEN=50#历史交互长度
PSEUDO=1000
NEIGHBOR_LEN=100#最多考虑的邻居节点个数
CLIP=0.1#梯度裁剪
LR=0.01#学习率
EPS=1#隐私预算
EPOCH=3#训练轮数
ACCUMULATION_STEPS = 4  # 梯度累积步数，用于模拟更大的批量大小