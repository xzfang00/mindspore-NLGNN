import mindspore.nn as nn
from mindspore.nn import ReLU
from mindspore_gl.nn import GCNConv
from mindspore_gl.nn import GATConv
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.nn import ReLU
from mindspore_gl.nn import GCNConv
from mindspore_gl.nn import GATConv
from mindspore.ops import operations as P
import mindspore
import time
import utils
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore import Tensor
import numpy as np
import mindspore.context as context
from mindspore_gl.nn import GATConv
from model import NLGNN
from mindspore.nn.optim import Adam

if __name__ == '__main__':
    print('-------------------------------------------------------' * 3)
    dataset_name = 'wisconsin'
    heter_dataset = ['chameleon', 'cornell', 'actor', 'squirrel', 'texas', 'wisconsin']  # 异配图
    homo_dataset = ['cora', 'citeseer', 'pubmed']  # 同配图
    print(f"此时数据集是{dataset_name}")
    lr = 0.01
    weight_decay = 5e-4
    max_epoch = 500
    patience = 200  # 如果超过这么多轮次没有提升，直接停
    num_hidden = [92, 48, 16, 8]
    dropout = 0
    le_list = ['mlp', 'gcn', 'gat']
    le = le_list[1]
    print(f"此时模型是{le}")
    device = 'GPU'  # 'CPU'
    context.set_context(mode=context.GRAPH_MODE, device_target=device)
    print("当前使用的设备是:", context.get_context("device_target"))
    window_size = 5

    split_by_label_flag = True  #划分数据集是否按照标签划分
    if dataset_name in ['chameleon', 'cornell', 'texas']:
        split_by_label_flag = False

    if dataset_name in heter_dataset:
        dataset, num_features, num_classes, edge_index = utils.load_heter_data(dataset_name)
    elif dataset_name in homo_dataset:
        dataset, num_features, num_classes, edge_index = utils.load_homo_data(dataset_name)
    else:
        print("我们现在没有{} dataset.".format(dataset_name))
    # 输出当前dataset的信息
    utils.graph_info(dataset, None)
    # 设置全局随机种子
    mindspore.set_seed(123)
    train_index, val_index, test_index = utils.split_data(dataset, 0.6, 0.2, 0.2, split_by_label_flag)
    utils.graph_info(dataset, train_index)
    # 得到dataset的x和label，x.shape为(num_node，num_feature)
    x, lable = utils.getxANDlable(dataset)
    x = x.astype(np.float32)
    x = Tensor(x)
    # 得到该dataset，对应的GraphField对象
    graphFiled, in_degree, out_degree = utils.getGraphFiled(x, edge_index=edge_index)  # x 是float32
    # net = NLGNN(graphFiled, x, le, window_size, num_features, num_hidden, num_classes, 0.5, in_degree, out_degree)
    # 权重衰减(Weight Decay)是一种正则化技术，用于防止神经网络过拟合。
    # 它具体来说，权重衰减会使得模型的参数更新变得更加缓慢，从而减少了模型对训练数据的过度拟合。
    # 系数越大，权重衰减的效果就越明显
    print(x.shape)

    first_1 = GATConv(in_feat_size=num_features, out_size=num_hidden[0], num_attn_head=1)
    first_2 = GATConv(in_feat_size=num_hidden[0], out_size=num_hidden[1],  num_attn_head=1)
    h = first_1(x,  *graphFiled.get_graph())
    print(h.shape)
    h = first_2(h,  *graphFiled.get_graph())
