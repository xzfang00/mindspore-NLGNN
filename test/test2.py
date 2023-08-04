import mindspore as ms
from mindspore_gl.dataset import CoraV2
from scipy import io
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import Graph


def get_edges_from_coo(adj_coo):
    # 获取非零元素的坐标
    row, col = adj_coo.nonzero()
    # 构造边索引数组
    edge_index = np.stack([row, col], axis=0)
    return edge_index


root = "../tmp/pubmed"
dataset = CoraV2(root, name='pubmed')
label = dataset.node_label
label = label.reshape((len(label), 1))
edge_index = get_edges_from_coo(dataset.adj_coo)
node_features = {"node_feat": node_feat, "label": label}
graph = Graph(edge_index, node_feat=node_features)
