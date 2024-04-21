import paddle
import paddle.sparse as sparse

# 创建CSR格式稀疏邻接矩阵
N = 5
edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
values = [1, 1, 1, 1, 1]

# 将edges拆分为行和列
rows = [edge[0] for edge in edges]
cols = [edge[1] for edge in edges]

# 构造crows (行偏移量)
crows = paddle.to_tensor([0, 1, 2, 3, 4, 5], dtype="int32")

adj = sparse.sparse_csr_tensor(
    crows,
    paddle.to_tensor(cols, dtype="int32"),
    paddle.to_tensor(values, dtype="float32"),
    [N, N],
)

# 节点特征
node_feat = paddle.randn([N, 16])


# GCN 层
class GCNLayer(paddle.nn.Layer):
    def __init__(self, in_feat, out_feat):
        super(GCNLayer, self).__init__()
        self.linear = paddle.nn.Linear(in_feat, out_feat)

    def forward(self, x, adj):
        x = self.linear(x)
        x = sparse.matmul(adj, x)
        return x


class GCN(paddle.nn.Layer):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(16, 32)
        self.relu = paddle.nn.ReLU()
        self.gcn2 = GCNLayer(32, 2)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.relu(h)
        h = self.gcn2(h, adj)
        return h


# 创建模型
model = GCN()

# 前向传播
logits = model(node_feat, adj)
