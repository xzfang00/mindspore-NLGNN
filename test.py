import mindspore as ms
from mindspore_gl.nn import GATConv
from mindspore_gl import GraphField
n_nodes = 4
n_edges = 7
feat_size = 4
src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
ones = ms.ops.Ones()
feat = ones((n_nodes, feat_size), ms.float32)
graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
gatconv = GATConv(in_feat_size=4, out_size=2, num_attn_head=3)
res = gatconv(feat, *graph_field.get_graph())
print(res.shape)

