#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
GDI-NN 精度对齐测试脚本

验证 PaddleMaterials 的 gdinn 实现与原始 GDI-NN (PyTorch) 实现的模型结构一致性:
- ppmat/models/gdinn/gnn.py <-> model/model_GNN.py
- ppmat/models/gdinn/mcm.py <-> model/model_MCM.py

通过对比相同随机输入的输出来验证模型结构一致性
"""

import os
import sys
import numpy as np
import pandas as pd
import paddle
import torch
import dgl

# 设置 Paddle 使用 GPU
paddle.set_device('gpu:0')

# 设置 PyTorch 使用 GPU (GDI-NN 代码内部硬编码了 .cuda())
torch.cuda.set_device(0)

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/shun/workspace/Projects/github/GDI-NN')

# ============================================================================
# 配置路径 (参考 quick_test.py)
# ============================================================================
GDI_NN_DIR = '/home/shun/workspace/Projects/github/GDI-NN'

class Config:
    # 数据目录配置
    DATASET_DIR = './test_gdinn/dataset'
    OUTPUT_DIR = './test_gdinn/data/alignment_test'

    # 数据文件
    SOLVENT_LIST_FILE = 'solvent_list.csv'
    BINARY_DATA_FILE = 'output_binary_with_inf_all.csv'
    
    # 精度阈值
    FORWARD_TOLERANCE = 1e-3
    
    # 测试参数
    BATCH_SIZE = 8
    HIDDEN_DIM = 64
    IN_DIM = 74  # Match GDI-NN's feature dimension
    NUM_CLASSES = 1

    @property
    def solvent_list_path(self):
        return os.path.join(self.DATASET_DIR, self.SOLVENT_LIST_FILE)

    @property
    def binary_data_path(self):
        return os.path.join(self.DATASET_DIR, self.BINARY_DATA_FILE)
    
    @property
    def output_dir(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        return self.OUTPUT_DIR


config = Config()


def set_random_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    paddle.seed(seed)
    torch.manual_seed(seed)


def torch_to_paddle_param(torch_param, transpose=False):
    """将 PyTorch 参数转换为 Paddle Tensor"""
    np_val = torch_param.detach().cpu().numpy()
    if transpose and np_val.ndim == 2:
        np_val = np_val.T  # PyTorch Linear: [out, in] -> Paddle Linear: [in, out]
    return paddle.to_tensor(np_val)


def sync_gnn_weights(torch_model, paddle_model):
    """将 PyTorch GNN 模型权重同步到 Paddle GNN 模型"""
    # GraphConv: weight [in, out], bias [out] — same layout
    paddle_model.conv1.weight.set_value(torch_to_paddle_param(torch_model.conv1.weight))
    paddle_model.conv1.bias.set_value(torch_to_paddle_param(torch_model.conv1.bias))
    paddle_model.conv2.weight.set_value(torch_to_paddle_param(torch_model.conv2.weight))
    paddle_model.conv2.bias.set_value(torch_to_paddle_param(torch_model.conv2.bias))

    # MPNNConv / MPNNconv
    tc = torch_model.global_conv1
    pc = paddle_model.global_conv

    # project_node_feats: Sequential(Linear, Activation)
    pc.project_node_feats[0].weight.set_value(torch_to_paddle_param(tc.project_node_feats[0].weight, transpose=True))
    pc.project_node_feats[0].bias.set_value(torch_to_paddle_param(tc.project_node_feats[0].bias))

    # gnn_layer.edge_func: Sequential(Linear, Activation, Linear)
    pc.gnn_layer.edge_func[0].weight.set_value(torch_to_paddle_param(tc.gnn_layer.edge_func[0].weight, transpose=True))
    pc.gnn_layer.edge_func[0].bias.set_value(torch_to_paddle_param(tc.gnn_layer.edge_func[0].bias))
    pc.gnn_layer.edge_func[2].weight.set_value(torch_to_paddle_param(tc.gnn_layer.edge_func[2].weight, transpose=True))
    pc.gnn_layer.edge_func[2].bias.set_value(torch_to_paddle_param(tc.gnn_layer.edge_func[2].bias))

    # gnn_layer.bias
    pc.gnn_layer.bias.set_value(torch_to_paddle_param(tc.gnn_layer.bias))

    # GRU (PyTorch) -> GRUCell (Paddle): same weight layout [3*hidden, input/hidden]
    pc.gru_cell.weight_ih.set_value(torch_to_paddle_param(tc.gru.weight_ih_l0))
    pc.gru_cell.weight_hh.set_value(torch_to_paddle_param(tc.gru.weight_hh_l0))
    pc.gru_cell.bias_ih.set_value(torch_to_paddle_param(tc.gru.bias_ih_l0))
    pc.gru_cell.bias_hh.set_value(torch_to_paddle_param(tc.gru.bias_hh_l0))

    # Classifier Linear layers: transpose weights
    paddle_model.classify1.weight.set_value(torch_to_paddle_param(torch_model.classify1.weight, transpose=True))
    paddle_model.classify1.bias.set_value(torch_to_paddle_param(torch_model.classify1.bias))
    paddle_model.classify2.weight.set_value(torch_to_paddle_param(torch_model.classify2.weight, transpose=True))
    paddle_model.classify2.bias.set_value(torch_to_paddle_param(torch_model.classify2.bias))
    paddle_model.classify3.weight.set_value(torch_to_paddle_param(torch_model.classify3.weight, transpose=True))
    paddle_model.classify3.bias.set_value(torch_to_paddle_param(torch_model.classify3.bias))


def sync_mcm_weights(torch_model, paddle_model):
    """将 PyTorch MCM 模型权重同步到 Paddle MCM 模型"""
    # Embedding: solvent_emb[0][0] -> solvent_emb.embedding
    torch_emb_seq = torch_model.solvent_emb[0]
    paddle_model.solvent_emb.embedding.weight.set_value(
        torch_to_paddle_param(torch_emb_seq[0].weight))

    # Linear layers: solvent_emb[0][3,6,9] -> solvent_emb.linear1,2,3
    paddle_model.solvent_emb.linear1.weight.set_value(
        torch_to_paddle_param(torch_emb_seq[3].weight, transpose=True))
    paddle_model.solvent_emb.linear1.bias.set_value(
        torch_to_paddle_param(torch_emb_seq[3].bias))
    paddle_model.solvent_emb.linear2.weight.set_value(
        torch_to_paddle_param(torch_emb_seq[6].weight, transpose=True))
    paddle_model.solvent_emb.linear2.bias.set_value(
        torch_to_paddle_param(torch_emb_seq[6].bias))
    paddle_model.solvent_emb.linear3.weight.set_value(
        torch_to_paddle_param(torch_emb_seq[9].weight, transpose=True))
    paddle_model.solvent_emb.linear3.bias.set_value(
        torch_to_paddle_param(torch_emb_seq[9].bias))

    # layers_end: two Sequential branches
    for branch_idx in range(2):
        torch_branch = torch_model.layers_end[branch_idx]
        paddle_branch = paddle_model.layers_end[branch_idx]
        # Copy all Linear layers
        for t_layer, p_layer in zip(torch_branch, paddle_branch):
            if hasattr(t_layer, 'weight') and hasattr(p_layer, 'weight'):
                p_layer.weight.set_value(torch_to_paddle_param(t_layer.weight, transpose=True))
                p_layer.bias.set_value(torch_to_paddle_param(t_layer.bias))


def prepare_data():
    """准备真实测试数据"""
    print("准备测试数据...")
    
    # 读取数据
    solvent_df = pd.read_csv(config.solvent_list_path)
    df = pd.read_csv(config.binary_data_path)
    
    # 取前500条数据进行测试
    df = df.head(500)
    
    print(f"✓ 加载数据: {len(df)} 样本, {len(solvent_df)} 溶剂")
    
    return df, solvent_df


# ============================================================================
# GNN 模型精度对齐测试
# ============================================================================

def test_gnn_alignment():
    """测试 GNN 模型精度对齐 (Paddle vs PyTorch)

    注意: 数据集自动计算 HB 特征（与 GDI-NN 原始行为一致）
    """
    print("\n" + "=" * 80)
    print("测试 GNN 精度对齐 (Paddle vs PyTorch)")
    print("=" * 80)
    
    try:        
        # 导入 Paddle 模型和数据集
        from ppmat.models.gdinn.gnn import SolvGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 导入 PyTorch 模型
        sys.path.insert(0, GDI_NN_DIR)
        from model.model_GNN import solvgnn_binary
        
        # 创建 Paddle 数据集（自动计算 HB 特征）
        paddle_dataset = BinaryActivityDataset(
            path=config.binary_data_path,
            solvent_list_path=config.solvent_list_path,
            add_self_loop=True,
            preload_graphs=False
        )
        
        # 创建采样器
        sampler = BatchSampler(
            dataset=paddle_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )
        
        collator = BinaryActivityCollator()
        paddle_loader = DataLoader(
            dataset=paddle_dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建 Paddle 模型
        paddle_model = SolvGNN(
            in_dim=config.IN_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_classes=config.NUM_CLASSES,
            num_step_message_passing=1,
            pinn_lambda=0.0
        )
        paddle_model.eval()
        
        # 创建 PyTorch 模型并移到 GPU
        torch_model = solvgnn_binary(
            in_dim=config.IN_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_classes=config.NUM_CLASSES,
            mlp_dropout_rate=0,
            mlp_activation="relu",
            mpnn_activation="relu",
            mlp_num_hid_layers=2
        )
        torch_model = torch_model.cuda()
        torch_model.eval()

        # 同步权重: PyTorch -> Paddle
        sync_gnn_weights(torch_model, paddle_model)

        # 获取一个 batch
        for batch_idx, paddle_batch in enumerate(paddle_loader):
            if batch_idx >= 1:
                break
            
            # 先获取 PyTorch 数据 (在 Paddle 前向传播之前，避免图特征被修改)
            g1_paddle = paddle_batch['g1']
            g2_paddle = paddle_batch['g2']
            x1_np = paddle_batch['x1'].numpy()

            # 提取 HB 特征 (GDI-NN 要求)
            inter_hb_np = paddle_batch['inter_hb'].numpy().flatten()
            intra_hb1_np = paddle_batch['intra_hb1'].numpy().flatten()
            intra_hb2_np = paddle_batch['intra_hb2'].numpy().flatten()

            # 转换 Paddle 图到 DGL 图
            g1_dgl = paddle_graph_to_dgl(g1_paddle)
            g2_dgl = paddle_graph_to_dgl(g2_paddle)

            # 创建 empty solvsys (匹配原始 generate_solvsys)
            empty_solvsys = create_empty_solvsys(config.BATCH_SIZE)

            # PyTorch 前向传播 (solv1_x 需要 1D tensor)
            torch_batch = {
                'g1': g1_dgl,
                'g2': g2_dgl,
                'solv1_x': torch.from_numpy(x1_np).flatten(),
                'inter_hb': torch.from_numpy(inter_hb_np),
                'intra_hb1': torch.from_numpy(intra_hb1_np),
                'intra_hb2': torch.from_numpy(intra_hb2_np),
            }
            
            with torch.no_grad():
                torch_output = torch_model(torch_batch, empty_solvsys, gamma_grad=False)
            
            torch_gamma1 = torch_output[:, 0].cpu().numpy()
            torch_gamma2 = torch_output[:, 1].cpu().numpy()
            
            # Paddle 前向传播
            with paddle.no_grad():
                paddle_output = paddle_model(paddle_batch)
            
            paddle_pred = paddle_output['pred_dict']
            # 使用 ln_gamma 进行比较 (PyTorch 输出的是 ln_gamma)
            paddle_gamma1 = paddle_pred['ln_gamma1'].numpy()
            paddle_gamma2 = paddle_pred['ln_gamma2'].numpy()
            
            # 比较
            diff_gamma1 = np.abs(paddle_gamma1.flatten() - torch_gamma1)
            diff_gamma2 = np.abs(paddle_gamma2.flatten() - torch_gamma2)
            
            max_diff_gamma1 = np.max(diff_gamma1)
            max_diff_gamma2 = np.max(diff_gamma2)
            mean_diff_gamma1 = np.mean(diff_gamma1)
            mean_diff_gamma2 = np.mean(diff_gamma2)
            
            print(f"  gamma1 最大差异: {max_diff_gamma1:.6f}")
            print(f"  gamma1 平均差异: {mean_diff_gamma1:.6f}")
            print(f"  gamma2 最大差异: {max_diff_gamma2:.6f}")
            print(f"  gamma2 平均差异: {mean_diff_gamma2:.6f}")
            
            passed = max(max_diff_gamma1, max_diff_gamma2) < config.FORWARD_TOLERANCE
            
            if passed:
                return True, "精度对齐"
            else:
                return False, f"gamma1 max diff: {max_diff_gamma1:.6f}, gamma2 max diff: {max_diff_gamma2:.6f}"
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False, f"导入失败: {e}"
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def paddle_graph_to_dgl(paddle_g):
    """将 Paddle batched 图转换为 DGL batched 图"""
    # 获取节点和边信息
    # pgl.Graph.edges is a tensor of shape [num_edges, 2]
    edges_tensor = paddle_g.edges
    # Handle both numpy arrays and paddle tensors
    if hasattr(edges_tensor, 'numpy'):
        src = edges_tensor[:, 0].numpy()
        dst = edges_tensor[:, 1].numpy()
    else:
        src = np.asarray(edges_tensor[:, 0])
        dst = np.asarray(edges_tensor[:, 1])
    num_nodes = int(paddle_g.num_nodes)
    
    # 获取节点特征 (Paddle 使用 node_feat 字典)
    if paddle_g.node_feat and 'h' in paddle_g.node_feat:
        h_feat = paddle_g.node_feat['h']
        node_feats = h_feat.numpy() if hasattr(h_feat, 'numpy') else np.asarray(h_feat)
    elif paddle_g.node_feat and 'feat' in paddle_g.node_feat:
        feat = paddle_g.node_feat['feat']
        node_feats = feat.numpy() if hasattr(feat, 'numpy') else np.asarray(feat)
    else:
        node_feats = np.random.randn(num_nodes, config.IN_DIM).astype(np.float32)
    
    # 创建 DGL 图并移到 GPU
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = g.to("cuda:0")  # 先把图移到 GPU
    g.ndata['h'] = torch.from_numpy(node_feats).cuda()
    
    # 设置 batch 信息 (如果存在)
    # pgl uses graph_node_id to identify which graph each node belongs to
    # Count nodes per graph
    graph_node_id_raw = paddle_g.graph_node_id
    graph_node_id = graph_node_id_raw.numpy() if hasattr(graph_node_id_raw, 'numpy') else np.asarray(graph_node_id_raw)
    num_graphs = int(paddle_g.num_graph)
    batch_num_nodes = np.array([np.sum(graph_node_id == i) for i in range(num_graphs)])
    g.set_batch_num_nodes(batch_num_nodes)
    
    # Count edges per graph
    graph_edge_id_raw = paddle_g.graph_edge_id
    graph_edge_id = graph_edge_id_raw.numpy() if hasattr(graph_edge_id_raw, 'numpy') else np.asarray(graph_edge_id_raw)
    batch_num_edges = np.array([np.sum(graph_edge_id == i) for i in range(num_graphs)])
    g.set_batch_num_edges(batch_num_edges)
    
    return g


def create_empty_solvsys(batch_size):
    """创建溶剂系统图 (GPU)，匹配原始 generate_solvsys 方法。

    原始代码:
        solvsys.add_nodes(n_solv * batch_size)
        src = arange(batch_size), dst = arange(batch_size, 2*batch_size)
        add_edges(cat(src, dst), cat(dst, src))  # bidirectional
        add_edges(arange(2*batch), arange(2*batch))  # self-loops
    """
    n_solv = 2
    num_nodes = n_solv * batch_size

    src_range = torch.arange(batch_size)
    dst_range = torch.arange(batch_size, num_nodes)
    all_range = torch.arange(num_nodes)

    # Bidirectional edges + self-loops
    edge_src = torch.cat([torch.cat([src_range, dst_range]), all_range])
    edge_dst = torch.cat([torch.cat([dst_range, src_range]), all_range])

    g = dgl.graph((edge_src, edge_dst), num_nodes=num_nodes)
    return g.to("cuda:0")


# ============================================================================
# MCM 模型精度对齐测试
# ============================================================================

def test_mcm_alignment():
    """测试 MCM 模型精度对齐 (Paddle vs PyTorch)"""
    print("\n" + "=" * 80)
    print("测试 MCM 精度对齐 (Paddle vs PyTorch)")
    print("=" * 80)
    
    try:
        # 准备数据
        df, solvent_df = prepare_data()
        
        # 导入 Paddle MCM 模型
        from ppmat.models.gdinn.mcm import MCM_MultiMLP
        
        # 导入 PyTorch MCM 模型
        sys.path.insert(0, GDI_NN_DIR)
        from model.model_MCM import MCM_multiMLP
        
        # 创建 Paddle MCM 模型
        solvent_id_max = len(solvent_df)
        
        paddle_model = MCM_MultiMLP(
            solvent_id_max=solvent_id_max,
            dim_hidden_channels=config.HIDDEN_DIM,
            dropout_hidden=0.0,
            dropout_interaction=0.0,
            mlp_num_hid_layers=1,
            pinn_lambda=0.0
        )
        paddle_model.eval()
        
        # 创建 PyTorch MCM 模型并移到 GPU
        torch_model = MCM_multiMLP(
            solvent_id_max=solvent_id_max,
            dim_hidden_channels=config.HIDDEN_DIM,
            dropout_hidden=0.0,
            dropout_interaction=0.0,
            mlp_num_hid_layers=1
        )
        torch_model = torch_model.cuda()
        torch_model.eval()

        # 同步权重: PyTorch -> Paddle
        sync_mcm_weights(torch_model, paddle_model)

        # 准备测试数据 (从真实数据中取)
        batch_size = min(config.BATCH_SIZE, len(df))
        
        # 查找溶剂 ID
        solv1_smiles = df['solv1'].values[:batch_size]
        solv2_smiles = df['solv2'].values[:batch_size]
        
        # 创建 SMILES 到 ID 的映射
        solvent_list = solvent_df['smiles_can'].tolist()
        solv1_id = [solvent_list.index(s) if s in solvent_list else 0 for s in solv1_smiles]
        solv2_id = [solvent_list.index(s) if s in solvent_list else 0 for s in solv2_smiles]
        
        x1 = df['solv1_x'].values[:batch_size].astype(np.float32)
        gamma1 = df['solv1_gamma'].values[:batch_size].astype(np.float32)
        gamma2 = df['solv2_gamma'].values[:batch_size].astype(np.float32)
        
        # Paddle 前向传播
        paddle_batch = {
            'solv1_id': paddle.to_tensor(solv1_id, dtype='int64'),
            'solv2_id': paddle.to_tensor(solv2_id, dtype='int64'),
            'x1': paddle.to_tensor(x1, dtype='float32'),
            'gamma1': paddle.to_tensor(gamma1, dtype='float32').reshape([-1, 1]),
            'gamma2': paddle.to_tensor(gamma2, dtype='float32').reshape([-1, 1]),
        }
        
        with paddle.no_grad():
            paddle_output = paddle_model(paddle_batch)
        
        paddle_pred = paddle_output['pred_dict']
        paddle_ln_gamma1 = paddle_pred['ln_gamma1'].numpy()
        paddle_ln_gamma2 = paddle_pred['ln_gamma2'].numpy()
        
        # PyTorch 前向传播 (solv1_x 需要 1D tensor)
        torch_batch = {
            'solv1_id': torch.tensor(solv1_id, dtype=torch.int64),
            'solv2_id': torch.tensor(solv2_id, dtype=torch.int64),
            'solv1_x': torch.tensor(x1, dtype=torch.float32).flatten(),
            'gamma1': torch.tensor(gamma1, dtype=torch.float32).reshape([-1, 1]),
            'gamma2': torch.tensor(gamma2, dtype=torch.float32).reshape([-1, 1]),
        }
        
        with torch.no_grad():
            torch_output = torch_model(torch_batch, None, gamma_grad=False)
        
        torch_ln_gamma1 = torch_output[:, 0].cpu().numpy()
        torch_ln_gamma2 = torch_output[:, 1].cpu().numpy()
        
        # 比较
        diff_ln_gamma1 = np.abs(paddle_ln_gamma1.flatten() - torch_ln_gamma1)
        diff_ln_gamma2 = np.abs(paddle_ln_gamma2.flatten() - torch_ln_gamma2)
        
        max_diff_gamma1 = np.max(diff_ln_gamma1)
        max_diff_gamma2 = np.max(diff_ln_gamma2)
        mean_diff_gamma1 = np.mean(diff_ln_gamma1)
        mean_diff_gamma2 = np.mean(diff_ln_gamma2)
        
        print(f"  ln_gamma1 最大差异: {max_diff_gamma1:.6f}")
        print(f"  ln_gamma1 平均差异: {mean_diff_gamma1:.6f}")
        print(f"  ln_gamma2 最大差异: {max_diff_gamma2:.6f}")
        print(f"  ln_gamma2 平均差异: {mean_diff_gamma2:.6f}")
        
        passed = max(max_diff_gamma1, max_diff_gamma2) < config.FORWARD_TOLERANCE
        
        if passed:
            return True, "精度对齐"
        else:
            return False, f"ln_gamma1 max diff: {max_diff_gamma1:.6f}, ln_gamma2 max diff: {max_diff_gamma2:.6f}"
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False, f"导入失败: {e}"
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


# ============================================================================
# Graph Utils 精度对齐测试
# ============================================================================

def test_mean_nodes_alignment():
    """测试 mean_nodes 函数精度对齐 (Paddle vs DGL)
    
    验证 PaddleMaterials 的 mean_nodes 实现与 DGL 的 dgl.mean_nodes 输出一致
    """
    print("\n" + "=" * 80)
    print("测试 mean_nodes 精度对齐 (Paddle vs DGL)")
    print("=" * 80)
    
    try:
        import pgl
        from ppmat.models.gdinn.utils.graph_utils import mean_nodes
        
        # 测试参数
        batch_size = 4
        feat_dim = 64
        num_nodes_list = [10, 15, 8, 12]  # 每个图的节点数不同
        
        # 设置随机种子
        set_random_seed(42)
        
        # 创建 Paddle 图列表
        paddle_graphs = []
        for i, num_nodes in enumerate(num_nodes_list):
            # 创建随机边
            num_edges = num_nodes * 3
            src = paddle.randint(0, num_nodes, [num_edges])
            dst = paddle.randint(0, num_nodes, [num_edges])
            
            # 创建随机节点特征
            node_feat = paddle.randn([num_nodes, feat_dim])
            
            # 创建 pgl.Graph
            edges = list(zip(src.tolist(), dst.tolist()))
            g = pgl.Graph(
                num_nodes=num_nodes,
                edges=edges,
                node_feat={'h': node_feat}
            )
            paddle_graphs.append(g)
        
        # 批处理 Paddle 图
        paddle_batched = pgl.Graph.batch(paddle_graphs)
        
        # 计算 Paddle mean_nodes
        paddle_result = mean_nodes(paddle_batched, 'h')
        
        print(f"  Paddle mean_nodes 输出形状: {paddle_result.shape}")
        
        # 创建对应的 DGL 图列表
        dgl_graphs = []
        for i, num_nodes in enumerate(num_nodes_list):
            # 使用相同的边
            # pgl.Graph.edges is a tensor of shape [num_edges, 2]
            edges_tensor = paddle_graphs[i].edges
            if hasattr(edges_tensor, 'numpy'):
                src_np = edges_tensor[:, 0].numpy()
                dst_np = edges_tensor[:, 1].numpy()
            else:
                src_np = np.asarray(edges_tensor[:, 0])
                dst_np = np.asarray(edges_tensor[:, 1])
            
            # 创建 DGL 图
            g = dgl.graph((src_np, dst_np), num_nodes=num_nodes)
            
            # 使用相同的节点特征
            h_feat = paddle_graphs[i].node_feat['h']
            node_feat_np = h_feat.numpy() if hasattr(h_feat, 'numpy') else np.asarray(h_feat)
            g.ndata['h'] = torch.from_numpy(node_feat_np)
            
            dgl_graphs.append(g)
        
        # 批处理 DGL 图
        dgl_batched = dgl.batch(dgl_graphs)
        
        # 计算 DGL mean_nodes
        dgl_result = dgl.mean_nodes(dgl_batched, 'h')
        
        print(f"  DGL mean_nodes 输出形状: {dgl_result.shape}")
        
        # 转换为 numpy 进行比较
        paddle_np = paddle_result.numpy()
        dgl_np = dgl_result.cpu().numpy()
        
        # 计算差异
        diff = np.abs(paddle_np - dgl_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  最大差异: {max_diff:.10f}")
        print(f"  平均差异: {mean_diff:.10f}")
        
        # 验证形状 (Paddle uses list, DGL uses torch.Size)
        assert list(paddle_result.shape) == list(dgl_result.shape), \
            f"形状不匹配: Paddle {paddle_result.shape} vs DGL {dgl_result.shape}"
        
        # 验证精度
        tolerance = 1e-6
        passed = max_diff < tolerance
        
        if passed:
            print(f"  ✓ 精度对齐 (差异 < {tolerance})")
            return True, "精度对齐"
        else:
            print(f"  ✗ 精度未对齐 (差异 {max_diff:.10f} >= {tolerance})")
            return False, f"最大差异: {max_diff:.10f}"
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False, f"导入失败: {e}"
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


# ============================================================================
# HB 特征测试
# ============================================================================

def test_hb_features():
    """测试氢键特征计算是否与原始 GDI-NN 一致
    
    原始 GDI-NN 实现 (generate_dataset_for_training.py):
        - solvent_data[solvent_id] = [graph, hba, hbd, min(hba, hbd)]
        - intra_hb1 = solv1[3]  # min(hba, hbd)
        - intra_hb2 = solv2[3]  # min(hba, hbd)
        - inter_hb = min(solv1[1], solv2[2]) + min(solv1[2], solv2[1])
                   = min(hba1, hbd2) + min(hbd1, hba2)
    """
    print("\n" + "=" * 80)
    print("测试氢键特征计算 (与原始 GDI-NN 对齐)")
    print("=" * 80)
    
    try:
        from ppmat.datasets import BinaryActivityDataset
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        # 创建数据集
        dataset = BinaryActivityDataset(
            path=config.binary_data_path,
            solvent_list_path=config.solvent_list_path,
            add_self_loop=True,
            preload_graphs=False
        )
        
        # 获取一个样本
        sample = dataset[0]
        
        # 验证 HB 特征存在
        assert 'intra_hb1' in sample, "缺少 intra_hb1 特征"
        assert 'intra_hb2' in sample, "缺少 intra_hb2 特征"
        assert 'inter_hb' in sample, "缺少 inter_hb 特征"
        
        print(f"✓ HB 特征字段存在")
        
        # 验证 HB 特征的计算值
        # 从溶剂列表获取 SMILES
        solvent_df = pd.read_csv(config.solvent_list_path)
        
        # 获取样本的溶剂 ID 和 SMILES
        solv1_id = sample['solv1_id']
        solv2_id = sample['solv2_id']
        
        solv1_row = solvent_df[solvent_df['solvent_id'] == solv1_id].iloc[0]
        solv2_row = solvent_df[solvent_df['solvent_id'] == solv2_id].iloc[0]
        
        smiles1 = solv1_row['smiles_can']
        smiles2 = solv2_row['smiles_can']
        
        # 使用 RDKit 计算 HBA 和 HBD (与原始 GDI-NN 一致)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        hba1 = rdMolDescriptors.CalcNumHBA(mol1)
        hbd1 = rdMolDescriptors.CalcNumHBD(mol1)
        hba2 = rdMolDescriptors.CalcNumHBA(mol2)
        hbd2 = rdMolDescriptors.CalcNumHBD(mol2)
        
        # 计算期望值 (原始 GDI-NN 公式)
        expected_intra_hb1 = min(hba1, hbd1)
        expected_intra_hb2 = min(hba2, hbd2)
        expected_inter_hb = min(hba1, hbd2) + min(hbd1, hba2)
        
        # 获取实际值
        actual_intra_hb1 = sample['intra_hb1']
        actual_intra_hb2 = sample['intra_hb2']
        actual_inter_hb = sample['inter_hb']
        
        print(f"\n溶剂 1: {solv1_id}")
        print(f"  SMILES: {smiles1}")
        print(f"  HBA: {hba1}, HBD: {hbd1}")
        print(f"  intra_hb1: expected={expected_intra_hb1}, actual={actual_intra_hb1}")
        
        print(f"\n溶剂 2: {solv2_id}")
        print(f"  SMILES: {smiles2}")
        print(f"  HBA: {hba2}, HBD: {hbd2}")
        print(f"  intra_hb2: expected={expected_intra_hb2}, actual={actual_intra_hb2}")
        
        print(f"\n交互氢键:")
        print(f"  inter_hb = min({hba1}, {hbd2}) + min({hbd1}, {hba2})")
        print(f"           = {min(hba1, hbd2)} + {min(hbd1, hba2)}")
        print(f"           = {expected_inter_hb}")
        print(f"  actual: {actual_inter_hb}")
        
        # 验证值是否匹配
        assert actual_intra_hb1 == expected_intra_hb1, \
            f"intra_hb1 不匹配: expected={expected_intra_hb1}, actual={actual_intra_hb1}"
        assert actual_intra_hb2 == expected_intra_hb2, \
            f"intra_hb2 不匹配: expected={expected_intra_hb2}, actual={actual_intra_hb2}"
        assert actual_inter_hb == expected_inter_hb, \
            f"inter_hb 不匹配: expected={expected_inter_hb}, actual={actual_inter_hb}"
        
        print(f"\n✓ HB 特征计算与原始 GDI-NN 一致")
        
        # 测试溶剂缓存机制
        print(f"\n测试溶剂缓存机制...")
        
        # 检查 solvent_data 缓存
        assert hasattr(dataset, 'solvent_data'), "数据集缺少 solvent_data 属性"
        
        # 验证缓存格式: [graph, hba, hbd, intra_hb]
        if solv1_id in dataset.solvent_data:
            cached = dataset.solvent_data[solv1_id]
            assert len(cached) == 4, f"缓存格式错误: 期望 4 个元素，实际 {len(cached)} 个"
            assert cached[1] == hba1, f"缓存 HBA 不匹配"
            assert cached[2] == hbd1, f"缓存 HBD 不匹配"
            assert cached[3] == min(hba1, hbd1), f"缓存 intra_hb 不匹配"
            print(f"  ✓ 溶剂缓存格式正确: [graph, hba={cached[1]}, hbd={cached[2]}, intra_hb={cached[3]}]")
        
        return True, "HB 特征测试通过"
        
    except AssertionError as e:
        print(f"\n✗ 断言失败: {e}")
        return False, str(e)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("GDI-NN 精度对齐测试 (使用真实数据)")
    print("=" * 80)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 运行测试
    results = {}
    
    # mean_nodes 精度对齐测试
    results['mean_nodes_精度对齐'] = test_mean_nodes_alignment()
    
    # HB 特征测试 (验证数据集实现与原始 GDI-NN 一致)
    results['HB_特征计算'] = test_hb_features()
    
    # GNN 精度对齐测试
    results['GNN_精度对齐'] = test_gnn_alignment()
    
    # MCM 精度对齐测试
    results['MCM_精度对齐'] = test_mcm_alignment()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, (result, message) in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status} ({message})")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    print("=" * 80)
    
    if failed == 0:
        print("✓ 所有测试通过!")
        return 0
    else:
        print(f"✗ {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
