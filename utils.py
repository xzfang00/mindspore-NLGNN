import mindspore
from mindspore.ops import operations as P
from mindspore_gl.dataset.cora import CoraV2
from scipy import io
import numpy as np
from mindspore.dataset import Graph
from mindspore_gl import GraphField


def get_edges_from_coo(adj_coo):  # 获取CoraV2类的边
    # 获取非零元素的坐标
    row, col = adj_coo.nonzero()
    # 构造边索引数组
    edge_index = np.stack([row, col], axis=0)
    return edge_index


def load_homo_data(dataset_name):
    root = "./tmp/"+dataset_name
    dataset = CoraV2(root, name=dataset_name)
    node_feat = dataset.node_feat
    label = dataset.node_label
    label = label.reshape((len(label), 1))
    edge_index = get_edges_from_coo(dataset.adj_coo)
    edge_index = to_undirected(edge_index)
    edge_index = remove_self_loops(edge_index)
    node_features = {"node_feat": node_feat, "label": label}
    graph = Graph(edge_index, node_feat=node_features)
    print(f"当前数据集的节点数{dataset.node_feat.shape[0]} 当前数据集的节点特征数{dataset.node_feat.shape[1]} 当前数据集的分类数{dataset.num_classes} 边数{edge_index.shape[1]}")
    print(f"边：{edge_index}")
    num_classes = int(dataset.num_classes)
    return graph, dataset.node_feat.shape[1], num_classes, edge_index


def load_heter_data(dataset_name):
    DATAPATH = 'data/heterophily_datasets_matlab'
    fulldata = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')
    edge_index = fulldata['edge_index']  # [[],[]]
    node_feat = fulldata['node_feat']  # （节点，每个节点特征向量）
    label = np.array(fulldata['label'], dtype=np.int32).flatten()  # flatten将向量铺平，铺成一维度的。
    num_features = node_feat.shape[1]  # 节点的特征数
    num_classes = np.max(label) + 1  # 分类数
    num_classes = int(num_classes)

    edges = to_undirected(edge_index)
    edges = remove_self_loops(edges)
    label = label.reshape((len(label), 1))
    node_features = {"node_feat": node_feat, "label": label}
    # 构造一个 Graph 对象
    graph = Graph(edges, node_feat=node_features)
    print(
        f"当前数据集的节点数{node_feat.shape[0]} 当前数据集的节点特征数{num_features} 当前数据集的分类数{num_classes} 边数：{edge_index.shape[1]}   ")
    print(f"边{edges}")
    return graph, num_features, num_classes, edges


def to_undirected(edge_index):
    # 将有向图转换为无向图
    # 获取边的数量
    num_edges = edge_index.shape[1]
    # 构造新的边索引数组
    undirected_edge_index = np.zeros((2, num_edges * 2), dtype=np.int64)
    # 填充新的边索引数组
    undirected_edge_index[:, :num_edges] = edge_index
    undirected_edge_index[:num_edges, num_edges:] = np.flip(edge_index, axis=0)
    return undirected_edge_index


def remove_self_loops(edge_index):  # 去除图种自环
    # 获取边的数量
    num_edges = edge_index.shape[1]

    # 找到所有非自环的边
    mask = edge_index[0] != edge_index[1]
    new_edge_index = edge_index[:, mask]

    return new_edge_index


def graph_info(dataset, index):
    print('-------------------------------------------------------'*3)
    graph = dataset.graph_info()
    print("graph info:", graph)
    if index is not None:
        nodes = dataset.get_all_nodes("0")[index]
    # 获取所有的节点信息
    else:
        nodes = dataset.get_all_nodes("0")
    nodes_list = nodes.tolist()

    # 获取特征和标签信息
    raw_tensor = dataset.get_node_feature(nodes_list, ['node_feat', 'label'])
    features, labels = raw_tensor[0], raw_tensor[1]

    # 打印特征和标签信息
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
    print("labels:", labels)
    print('--------------------------------------------------------'*3)


def split_data(graph, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, split_by_label_flag =True):
    # 初始化训练集、验证集和测试集
    train_index, val_index, test_index = [], [], []
    # 获取所有节点的标签信息
    nodes = graph.get_all_nodes("0")
    nodes_list = nodes.tolist()
    raw_tensor = graph.get_node_feature(nodes_list, ['node_feat', 'label'])
    # 按照标签分组
    if split_by_label_flag:
        labels = raw_tensor[1]
        num_classes = np.max(labels) + 1
        for i in range(num_classes):
            # 获取当前标签对应的节点索引
            index = np.where(labels == i)[0]
            # 随机打乱节点索引
            index = np.random.permutation(index)
            # 按照比例划分数据集
            n = len(index)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_index.append(index[:n_train])
            val_index.append(index[n_train:n_train + n_val])
            test_index.append(index[n_train + n_val:])
        # 合并数据集
        train_index = np.concatenate(train_index)
        val_index = np.concatenate(val_index)
        test_index = np.concatenate(test_index)
    else:
        features = raw_tensor[0]
        num_nodes = features.shape[0] # 总的节点数
        # 随机打乱节点索引
        index = np.random.permutation(num_nodes)
        # 按照比例划分数据集
        n_train = int(num_nodes * train_ratio)
        n_val = int(num_nodes * val_ratio)
        train_index = index[:n_train]
        val_index = index[n_train:n_train + n_val]
        test_index = index[n_train + n_val:]

    return train_index, val_index, test_index


def getxANDlable(graph):
    nodes = graph.get_all_nodes('0')
    nodes_list = nodes.tolist()
    raw_tensor = graph.get_node_feature(nodes_list, ['node_feat', 'label'])
    features, labels = raw_tensor[0], raw_tensor[1]
    return features, labels


def getGraphFiled(x, edge_index):
    src_idx = mindspore.Tensor(edge_index[0], mindspore.int32)
    dst_idx = mindspore.Tensor(edge_index[1], mindspore.int32)
    nodes_num = x.shape[0]
    edges_num = edge_index.shape[1]
    graph_field = GraphField(src_idx, dst_idx, nodes_num, edges_num)
    n_nodes = max(max(src_idx), max(dst_idx)) + 1

    # 初始化入度和出度数组
    in_degree = np.zeros(n_nodes, dtype=int)
    out_degree = np.zeros(n_nodes, dtype=int)
    # 计算每个节点的入度和出度
    for src, dst in zip(src_idx, dst_idx):
        out_degree[src] += 1
        in_degree[dst] += 1

    out_degree = mindspore.Tensor(out_degree, mindspore.int32)
    in_degree = mindspore.Tensor(in_degree, mindspore.int32)
    return graph_field, in_degree, out_degree