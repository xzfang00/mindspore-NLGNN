import mindspore.nn as nn
from mindspore.nn import ReLU
from mindspore_gl.nn import GCNConv
from mindspore_gl.nn import GATConv
from mindspore.ops import operations as P


class NLGNN(nn.Cell):
    def __init__(self, le, window_size, num_features, num_hidden, num_classes, dropout, graph, in_degree, out_degree, x):
        super().__init__()
        self.x = x
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.le = le
        self.graph = graph
        self.attention_layer = nn.Dense(num_hidden[1], 1)
        self.window_size = window_size
        self.conv1d1 = nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, pad_mode='same')
        self.conv1d2 = nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, pad_mode='same')
        self.final_layer = nn.Dense(2 * num_hidden[1], num_classes)
        self.dropout = dropout
        if le == 'mlp':
            self.first_1 = nn.Dense(int(num_features), num_hidden[0])
            self.first_2 = nn.Dense(num_hidden[0], num_hidden[1])
        elif le == 'gcn':
            self.first_1 = GCNConv(num_features, num_hidden[0], ReLU(), 0.5)
            self.first_2 = GCNConv(num_hidden[0], num_hidden[1], ReLU())
        else: # 'gat'
            self.first_1 = GATConv(num_features, num_hidden[0], num_attn_head=1, activation=ReLU())
            self.first_2 = GATConv(num_hidden[0], num_hidden[1], num_attn_head=1, activation=ReLU())

    def construct(self):
        x = self.x
        graph = self.graph
        in_degree = self.in_degree
        out_degree = self.out_degree
        if self.le == 'mlp':
            h = self.first_1(x)
            relu = nn.ReLU()
            dropout = nn.Dropout(p=0.5)
            h = relu(h)
            h = dropout(h)
            h = self.first_2(h)

        elif self.le == 'gcn':  # gcn
            h = self.first_1(x, in_degree, out_degree, *graph)
            h = self.first_2(h, in_degree, out_degree, *graph)

        else:  # gat
            h = self.first_1(x, *graph)
            h = self.first_2(h, *graph)

        before_h = h
        a = self.attention_layer(h)  # 将节点特征变成一个数，可看出权重
        h = a * h  # 各个点的特征*自己对于的权重值
        sort_index = P.TopK(sorted=True)(a.flatten(), a.shape[0])[1]  # 对权重由大到小排序
        restore_index = sort_index.argsort()  # 排序后的你序列
        h = P.Transpose()(h[sort_index], (1, 0)).unsqueeze(0)  # 排序再转置，在0的位置添加1个维度
        h = self.conv1d1(h)  # h此时shape(feature, nodes),可看成提取重要节点的信息。
        relu = nn.ReLU()
        dropout = nn.Dropout(p=0.5)
        h = relu(h)
        h = dropout(h)
        h = self.conv1d2(h)
        h = P.Squeeze()(h).T    # 把为1得维度去掉，然后转置
        h = h[restore_index]  # 排序后，再换回来

        final_h = P.Concat(axis=1)((before_h, h))  # 拼接之前h与现在h的特征为最后一层的输出
        final_h = self.final_layer(final_h)
        # 返回所有节点各个分类的概率。
        return P.LogSoftmax(axis=1)(final_h)
