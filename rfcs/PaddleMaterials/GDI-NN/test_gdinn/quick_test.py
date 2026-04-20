#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
GDI-NN 快速测试脚本
验证训练和预测流程是否正常工作
"""

import os
import sys
import paddle
import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppmat'))

# ============================================================================
# 配置路径
# ============================================================================
class Config:
    # 数据目录配置
    DATASET_DIR = './test_gdinn/dataset'
    OUTPUT_DIR = './test_gdinn/data/gdinn'

    # 原始数据文件
    SOLVENT_LIST_FILE = 'solvent_list.csv'
    BINARY_DATA_FILE = 'output_binary_with_inf_all.csv'

    # 输出数据文件
    TRAIN_BINARY_FILE = 'train_binary.csv'
    VAL_BINARY_FILE = 'val_binary.csv'
    TEST_BINARY_FILE = 'test_binary.csv'
    SOLVENT_LIST_OUTPUT = 'solvent_list.csv'

    # MCM 数据文件
    TRAIN_MCM_FILE = 'train_mcm.csv'
    VAL_MCM_FILE = 'val_mcm.csv'
    TEST_MCM_FILE = 'test_mcm.csv'

    # 数据分割比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    # TEST_RATIO = 0.1

    # 完整路径
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

    @property
    def train_binary_path(self):
        return os.path.join(self.OUTPUT_DIR, self.TRAIN_BINARY_FILE)

    @property
    def val_binary_path(self):
        return os.path.join(self.OUTPUT_DIR, self.VAL_BINARY_FILE)

    @property
    def test_binary_path(self):
        return os.path.join(self.OUTPUT_DIR, self.TEST_BINARY_FILE)

    @property
    def solvent_list_output_path(self):
        return os.path.join(self.OUTPUT_DIR, self.SOLVENT_LIST_OUTPUT)

    @property
    def train_mcm_path(self):
        return os.path.join(self.OUTPUT_DIR, self.TRAIN_MCM_FILE)

    @property
    def val_mcm_path(self):
        return os.path.join(self.OUTPUT_DIR, self.VAL_MCM_FILE)

    @property
    def test_mcm_path(self):
        return os.path.join(self.OUTPUT_DIR, self.TEST_MCM_FILE)


# 全局配置实例
config = Config()


def create_test_data():
    """创建测试数据（使用 GDI-NN 格式）"""
    import pandas as pd

    print("准备测试数据（使用 GDI-NN 格式）...")

    # 使用配置路径
    solvent_list_path = config.solvent_list_path
    output_binary_path = config.binary_data_path
    output_dir = config.output_dir

    # 复制溶剂列表（直接使用）
    print(f"读取溶剂列表: {solvent_list_path}")
    solvent_df = pd.read_csv(solvent_list_path)
    solvent_df.to_csv(config.solvent_list_output_path, index=False)
    print(f"✓ 溶剂数量: {len(solvent_df)}")

    # 读取 GDI-NN 格式数据
    print(f"读取数据: {output_binary_path}")
    df = pd.read_csv(output_binary_path)
    print(f"✓ 数据量: {len(df)}")

    # GDI-NN 格式：solv1_gamma, solv2_gamma 存储 ln_gamma
    # 取前5000条数据进行测试（如果数据量足够）
    df = df.head(5000)

    # 分割数据集（使用比例）
    n = len(df)
    train_size = int(n * config.TRAIN_RATIO)
    val_size = int(n * config.VAL_RATIO)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # 保存数据（GDI-NN 格式）
    train_df.to_csv(config.train_binary_path, index=False)
    val_df.to_csv(config.val_binary_path, index=False)
    test_df.to_csv(config.test_binary_path, index=False)

    print(f"✓ 训练集: {len(train_df)} 样本")
    print(f"✓ 验证集: {len(val_df)} 样本")
    print(f"✓ 测试集: {len(test_df)} 样本")
    print(f"✓ 数据保存在: {config.OUTPUT_DIR}/")


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("测试数据加载")
    print("=" * 80)

    try:
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator

        # 创建数据集（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )
        
        print(f"✓ 数据集创建成功")
        print(f"  样本数量: {len(dataset)}")
        
        # 创建采样器
        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        
        # 创建数据加载器
        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"  Batch数量: {len(dataloader)}")
        
        # 测试加载数据
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # 只测试前2个batch
                break
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  g1 nodes: {batch['g1'].num_nodes}")
            print(f"  g1 edges: {batch['g1'].num_edges}")
            print(f"  g2 nodes: {batch['g2'].num_nodes}")
            print(f"  g2 edges: {batch['g2'].num_edges}")
            print(f"  empty_solvsys nodes: {batch['empty_solvsys'].num_nodes}")
            print(f"  empty_solvsys edges: {batch['empty_solvsys'].num_edges}")
            print(f"  x1 shape: {batch['x1'].shape}")
            print(f"  x2 shape: {batch['x2'].shape}")
            print(f"  gamma1 shape: {batch['gamma1'].shape}")
            print(f"  gamma2 shape: {batch['gamma2'].shape}")
        
        print("\n✓ 数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 80)
    print("测试模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型 (使用与原始 GDI-NN 一致的默认参数)
        model = SolvGNN(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 测试前向传播
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 前向传播
            output = model(batch)
            
            print(f"✓ 前向传播成功")
            print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
            print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
            print(f"  loss: {output['loss_dict']['loss'].item():.4f}")
            print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
            print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
            print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
            print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
        
        print("\n✓ 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n" + "=" * 80)
    print("测试训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.losses import GibbsDuhemLoss
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型 (使用与原始 GDI-NN 一致的默认参数)
        model = SolvGNN(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        # 创建损失函数
        criterion = GibbsDuhemLoss()

        print(f"✓ 模型和损失函数创建成功")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        
        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            # 前向传播
            output = model(batch)
            loss = output['loss_dict']['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        print("\n✓ 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solvgnn_xmlp_forward():
    """测试 SolvGNNxMLP 模型前向传播"""
    print("\n" + "=" * 80)
    print("测试 SolvGNNxMLP 模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNNxMLP
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型
        model = SolvGNNxMLP(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            mlp_num_hid_layers=2,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ SolvGNNxMLP 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 测试前向传播
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 前向传播
            output = model(batch)
            
            print(f"✓ 前向传播成功")
            print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
            print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
            print(f"  loss: {output['loss_dict']['loss'].item():.4f}")
            print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
            print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
            print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
            print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
        
        print("\n✓ SolvGNNxMLP 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ SolvGNNxMLP 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solvgnn_xmlp_training():
    """测试 SolvGNNxMLP 模型训练步骤"""
    print("\n" + "=" * 80)
    print("测试 SolvGNNxMLP 模型训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNNxMLP
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型
        model = SolvGNNxMLP(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            mlp_num_hid_layers=2,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ SolvGNNxMLP 模型创建成功")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            # 前向传播
            output = model(batch)
            loss = output['loss_dict']['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        print("\n✓ SolvGNNxMLP 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ SolvGNNxMLP 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gegnn_forward():
    """测试 GEGNN 模型前向传播"""
    print("\n" + "=" * 80)
    print("测试 GEGNN 模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models import GEGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型
        model = GEGNN(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ GEGNN 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 测试前向传播
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 前向传播
            output = model(batch)
            
            print(f"✓ 前向传播成功")
            print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
            print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
            print(f"  loss: {output['loss_dict']['loss'].item():.4f}")
            print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
            print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
            print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
            print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
            if 'G_E' in output['pred_dict']:
                print(f"  G_E shape: {output['pred_dict']['G_E'].shape}")
                print(f"  G_E mean: {output['pred_dict']['G_E'].mean().item():.4f}")
        
        print("\n✓ GEGNN 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ GEGNN 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gegnn_training():
    """测试 GEGNN 模型训练步骤"""
    print("\n" + "=" * 80)
    print("测试 GEGNN 模型训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models import GEGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型
        model = GEGNN(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        print(f"✓ GEGNN 模型创建成功")

        # 创建数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )

        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            # 前向传播
            output = model(batch)
            loss = output['loss_dict']['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}, "
                  f"G_E = {output['pred_dict']['G_E'].mean().item():.4f}")
        
        print("\n✓ GEGNN 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ GEGNN 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_data_mcm():
    """创建 MCM 模型测试数据（使用 ID 而非 SMILES）"""
    import pandas as pd
    
    print("创建 MCM 测试数据...")
    
    # 创建简单的测试数据
    data = []
    
    # 使用 ID 而非 SMILES
    # 假设有 10 种不同的溶剂
    solvent_pairs = [
        # Solvent ID pairs (solv1_id, solv2_id, temp, x1, gamma1, gamma2)
        (0, 1, 298.15, 0.5, 1.2, 0.8),
        (1, 2, 298.15, 0.5, 1.1, 0.9),
        (0, 2, 298.15, 0.5, 1.3, 0.7),
        (2, 3, 298.15, 0.5, 1.15, 0.85),
        (3, 4, 298.15, 0.5, 1.25, 0.75),
    ]
    
    # 重复生成更多数据
    for solv1_id, solv2_id, temp, x1, gamma1, gamma2 in solvent_pairs:
        for _ in range(100):  # 每个组合生成100个样本
            # 添加一些随机变化
            x1_var = np.clip(x1 + np.random.normal(0, 0.1), 0.01, 0.99)
            x2 = 1.0 - x1_var
            
            # 简单的活度系数模拟
            gamma1_var = gamma1 * (1 + 0.1 * np.random.randn())
            gamma2_var = gamma2 * (1 + 0.1 * np.random.randn())
            
            # 转换为 ln_gamma
            ln_gamma1_var = np.log(abs(gamma1_var))
            ln_gamma2_var = np.log(abs(gamma2_var))
            
            data.append({
                'solv1_id': solv1_id,
                'solv2_id': solv2_id,
                'temperature (K)': temp + np.random.normal(0, 5),
                'x(1)': x1_var,
                'x(2)': x2,
                'ln_gamma_1': ln_gamma1_var,
                'ln_gamma_2': ln_gamma2_var
            })
    
    # 创建目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 保存数据
    df = pd.DataFrame(data)

    # 分割数据集
    train_df = df.iloc[:400]
    val_df = df.iloc[400:450]
    test_df = df.iloc[450:]

    train_df.to_csv(config.train_mcm_path, index=False)
    val_df.to_csv(config.val_mcm_path, index=False)
    test_df.to_csv(config.test_mcm_path, index=False)

    print(f"✓ MCM 训练集: {len(train_df)} 样本")
    print(f"✓ MCM 验证集: {len(val_df)} 样本")
    print(f"✓ MCM 测试集: {len(test_df)} 样本")
    print(f"✓ 数据保存在: {config.OUTPUT_DIR}/")
    
    return 5  # 返回最大 solvent ID


def test_mcm_forward():
    """测试 MCM 模型前向传播"""
    print("\n" + "=" * 80)
    print("测试 MCM 模型前向传播")
    print("=" * 80)
    
    try:
        from ppmat.models.gdinn.mcm import MCM_MultiMLP
        import pandas as pd
        
        # 创建测试数据
        max_solvent_id = create_test_data_mcm()
        
        # 创建模型
        model = MCM_MultiMLP(
            solvent_id_max=max_solvent_id,
            dim_hidden_channels=64,
            dropout_hidden=0.05,
            dropout_interaction=0.03,
            mlp_num_hid_layers=1,
            pinn_lambda=1.0
        )
        
        print(f"✓ MCM 模型创建成功")
        param_count = sum(p.numel().item() for p in model.parameters())
        print(f"  参数数量: {param_count}")
        
        # 加载测试数据
        df = pd.read_csv(config.train_mcm_path)
        
        # 创建 batch data
        batch_size = 32
        batch_data = {
            'solv1_id': paddle.to_tensor(df['solv1_id'].values[:batch_size], dtype='int64'),
            'solv2_id': paddle.to_tensor(df['solv2_id'].values[:batch_size], dtype='int64'),
            'x1': paddle.to_tensor(df['x(1)'].values[:batch_size], dtype='float32'),
            'gamma1': paddle.to_tensor(df['ln_gamma_1'].values[:batch_size], dtype='float32').unsqueeze(-1),
            'gamma2': paddle.to_tensor(df['ln_gamma_2'].values[:batch_size], dtype='float32').unsqueeze(-1)
        }
        
        print(f"\n测试前向传播...")
        
        # 前向传播
        output = model(batch_data)
        
        print(f"✓ 前向传播成功")
        print(f"  loss_dict keys: {list(output['loss_dict'].keys())}")
        print(f"  pred_dict keys: {list(output['pred_dict'].keys())}")
        print(f"  pred_loss: {output['loss_dict'].get('pred_loss', 0).item():.4f}")
        print(f"  gd_loss: {output['loss_dict'].get('gd_loss', 0).item():.4f}")
        print(f"  gamma1 shape: {output['pred_dict']['gamma1'].shape}")
        print(f"  gamma2 shape: {output['pred_dict']['gamma2'].shape}")
        
        print("\n✓ MCM 模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ MCM 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcm_training():
    """测试 MCM 模型训练步骤"""
    print("\n" + "=" * 80)
    print("测试 MCM 模型训练步骤")
    print("=" * 80)
    
    try:
        from ppmat.models.gdinn.mcm import MCM_MultiMLP
        import pandas as pd
        
        # 创建测试数据
        max_solvent_id = 5
        
        # 创建模型
        model = MCM_MultiMLP(
            solvent_id_max=max_solvent_id,
            dim_hidden_channels=64,
            dropout_hidden=0.05,
            dropout_interaction=0.03,
            mlp_num_hid_layers=1,
            pinn_lambda=1.0
        )
        
        print(f"✓ MCM 模型创建成功")
        
        # 加载训练数据
        df = pd.read_csv(config.train_mcm_path)
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试训练步骤
        model.train()
        
        batch_size = 32
        num_batches = min(3, len(df) // batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # 创建 batch data
            batch_data = {
                'solv1_id': paddle.to_tensor(df['solv1_id'].values[start_idx:end_idx], dtype='int64'),
                'solv2_id': paddle.to_tensor(df['solv2_id'].values[start_idx:end_idx], dtype='int64'),
                'x1': paddle.to_tensor(df['x(1)'].values[start_idx:end_idx], dtype='float32'),
                'gamma1': paddle.to_tensor(df['ln_gamma_1'].values[start_idx:end_idx], dtype='float32').unsqueeze(-1),
                'gamma2': paddle.to_tensor(df['ln_gamma_2'].values[start_idx:end_idx], dtype='float32').unsqueeze(-1)
            }
            
            # 前向传播
            output = model(batch_data)
            loss = output['loss_dict']['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"  Step {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        print("\n✓ MCM 训练步骤测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ MCM 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gibbs_duhem_loss():
    """测试 GibbsDuhemLoss 损失函数"""
    print("\n" + "=" * 80)
    print("测试 GibbsDuhemLoss 损失函数")
    print("=" * 80)
    
    try:
        from ppmat.losses import GibbsDuhemLoss
        
        # 创建损失函数实例
        criterion = GibbsDuhemLoss(lambda_gd=1.0, loss_type='mse')
        print(f"✓ GibbsDuhemLoss 创建成功")
        print(f"  lambda_gd: {criterion.lambda_gd}")
        print(f"  loss_type: {criterion.loss_type}")
        
        # 测试 1: 简单的合成数据测试
        print("\n测试 1: 合成数据测试")
        batch_size = 10
        
        # 创建测试数据
        # x1 在 [0.1, 0.9] 范围内
        x1 = paddle.linspace(0.1, 0.9, batch_size).unsqueeze(-1)
        x1.stop_gradient = False
        
        # 创建简单的 ln_gamma 函数（满足 Gibbs-Duhem 约束）
        # 例如：ln_gamma1 = A * x2^2, ln_gamma2 = A * x1^2
        # 这满足 x1 * d(ln_gamma1)/dx1 + x2 * d(ln_gamma2)/dx1 = 0
        A = 2.0
        x2 = 1 - x1
        ln_gamma1 = A * x2 * x2
        ln_gamma2 = A * x1 * x1
        
        # 计算损失
        loss = criterion(ln_gamma1, ln_gamma2, x1)
        print(f"  满足约束的损失: {loss.item():.6f}")
        
        # 测试 2: 不满足约束的情况
        print("\n测试 2: 不满足约束的情况")
        ln_gamma1_bad = paddle.randn([batch_size, 1])
        ln_gamma2_bad = paddle.randn([batch_size, 1])
        x1_test = paddle.linspace(0.1, 0.9, batch_size).unsqueeze(-1)
        x1_test.stop_gradient = False
        
        # 需要重新计算以建立计算图
        ln_gamma1_bad = x1_test * 3.0  # 简单的线性函数
        ln_gamma2_bad = x1_test * 2.0
        
        loss_bad = criterion(ln_gamma1_bad, ln_gamma2_bad, x1_test)
        print(f"  不满足约束的损失: {loss_bad.item():.6f}")
        
        # 测试 3: 不同的损失类型
        print("\n测试 3: 测试不同损失类型")
        for loss_type in ['mse', 'mae', 'huber']:
            criterion_type = GibbsDuhemLoss(lambda_gd=1.0, loss_type=loss_type)
            
            x1_type = paddle.linspace(0.1, 0.9, batch_size).unsqueeze(-1)
            x1_type.stop_gradient = False
            x2_type = 1 - x1_type
            
            # 使用满足约束的函数
            ln_gamma1_type = 2.0 * x2_type * x2_type
            ln_gamma2_type = 2.0 * x1_type * x1_type
            
            loss_type_val = criterion_type(ln_gamma1_type, ln_gamma2_type, x1_type)
            print(f"  {loss_type} 损失: {loss_type_val.item():.6f}")
        
        # 测试 5: 梯度计算测试
        print("\n测试 5: 梯度计算测试")
        criterion_grad = GibbsDuhemLoss(lambda_gd=1.0, create_graph=True)
        
        x1_grad = paddle.linspace(0.1, 0.9, batch_size).unsqueeze(-1)
        x1_grad.stop_gradient = False
        
        # 创建可训练参数
        A_param = paddle.create_parameter(shape=[1], dtype='float32', default_initializer=paddle.nn.initializer.Constant(2.0))
        
        x2_grad = 1 - x1_grad
        ln_gamma1_grad = A_param * x2_grad * x2_grad
        ln_gamma2_grad = A_param * x1_grad * x1_grad
        
        loss_grad = criterion_grad(ln_gamma1_grad, ln_gamma2_grad, x1_grad)
        
        # 计算梯度
        loss_grad.backward()
        
        print(f"  损失值: {loss_grad.item():.6f}")
        print(f"  A_param 梯度: {A_param.grad.item():.6f}")
        
        print("\n✓ GibbsDuhemLoss 测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ GibbsDuhemLoss 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gibbs_duhem_loss_with_model():
    """测试 GibbsDuhemLoss 与模型集成"""
    print("\n" + "=" * 80)
    print("测试 GibbsDuhemLoss 与模型集成")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.losses import GibbsDuhemLoss
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型
        model = SolvGNN(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=0.0  # 禁用模型内部的 GD loss
        )
        
        # 创建独立的 GibbsDuhemLoss
        criterion_gd = GibbsDuhemLoss(lambda_gd=1.0, loss_type='mse')
        
        print(f"✓ 模型和 GibbsDuhemLoss 创建成功")
        
        # 创建数据加载器
        dataset = BinaryActivityDataset(
            data_path=config.train_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )
        
        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )
        
        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )
        
        # 创建优化器
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=0.001
        )
        
        print(f"✓ 优化器创建成功")
        
        # 测试使用 GibbsDuhemLoss 计算额外损失
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            print(f"\n测试 Batch {batch_idx + 1}...")
            
            # 准备输入数据，确保 x1 可以计算梯度
            x1_gd = batch['x1'].clone()
            x1_gd.stop_gradient = False
            
            # 前向传播
            output = model(batch)
            
            # 获取预测值
            gamma1_pred = output['pred_dict']['gamma1']
            gamma2_pred = output['pred_dict']['gamma2']
            
            # 计算预测损失 (MSE)
            pred_loss = paddle.nn.functional.mse_loss(gamma1_pred, batch['gamma1']) + \
                       paddle.nn.functional.mse_loss(gamma2_pred, batch['gamma2'])
            
            # 使用 criterion_gd 计算 Gibbs-Duhem 约束损失
            gd_loss = criterion_gd(gamma1_pred, gamma2_pred, x1_gd)
            
            # 总损失 = 预测损失 + Gibbs-Duhem 约束损失
            total_loss = pred_loss + gd_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            optimizer.clear_grad()
            
            print(f"✓ 训练步骤成功")
            print(f"  pred_loss: {pred_loss.item():.4f}")
            print(f"  gd_loss (使用 criterion_gd): {gd_loss.item():.4f}")
            print(f"  total_loss: {total_loss.item():.4f}")
            print(f"  gamma1 shape: {gamma1_pred.shape}")
            print(f"  gamma2 shape: {gamma2_pred.shape}")
        
        print("\n✓ GibbsDuhemLoss 与模型集成测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ GibbsDuhemLoss 与模型集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """测试预测"""
    print("\n" + "=" * 80)
    print("测试预测")
    print("=" * 80)
    
    try:
        from ppmat.models import SolvGNN
        from ppmat.datasets import BinaryActivityDataset
        from paddle.io import DataLoader, BatchSampler
        from ppmat.datasets.collate_fn import BinaryActivityCollator
        
        # 创建模型 (使用与原始 GDI-NN 一致的默认参数)
        model = SolvGNN(
            in_dim=74,  # Match GDI-NN's feature dimension
            hidden_dim=64,
            n_classes=1,
            num_step_message_passing=1,
            pinn_lambda=1.0
        )

        # 创建测试数据加载器（GDI-NN 格式）
        dataset = BinaryActivityDataset(
            data_path=config.test_binary_path,
            solvent_list_path=config.solvent_list_output_path,
            add_self_loop=True,
            preload_graphs=False
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False
        )

        collator = BinaryActivityCollator()
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collator
        )

        # 测试预测
        model.eval()
        
        all_predictions = []
        
        with paddle.no_grad():
            for batch in dataloader:
                output = model(batch)
                pred_dict = output['pred_dict']
                
                gamma1_pred = pred_dict['gamma1'].numpy()
                gamma2_pred = pred_dict['gamma2'].numpy()
                gamma1_target = batch['gamma1'].numpy()
                gamma2_target = batch['gamma2'].numpy()
                
                for i in range(len(gamma1_pred)):
                    all_predictions.append({
                        'gamma1_pred': gamma1_pred[i],
                        'gamma2_pred': gamma2_pred[i],
                        'gamma1_target': gamma1_target[i],
                        'gamma2_target': gamma2_target[i]
                    })
        
        print(f"✓ 预测完成")
        print(f"  预测样本数: {len(all_predictions)}")
        
        # 计算简单的误差
        mae = np.mean([
            abs(p['gamma1_pred'] - p['gamma1_target']) +
            abs(p['gamma2_pred'] - p['gamma2_target'])
            for p in all_predictions
        ]) / 2
        
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        
        # 显示前5个预测结果
        print(f"\n前5个预测结果:")
        for i, pred in enumerate(all_predictions[:5]):
            print(f"  {i+1}. gamma1: pred={float(pred['gamma1_pred']):.4f}, target={float(pred['gamma1_target']):.4f}, "
                  f"error={abs(float(pred['gamma1_pred']) - float(pred['gamma1_target'])):.4f}")
            print(f"     gamma2: pred={float(pred['gamma2_pred']):.4f}, target={float(pred['gamma2_target']):.4f}, "
                  f"error={abs(float(pred['gamma2_pred']) - float(pred['gamma2_target'])):.4f}")
        
        print("\n✓ 预测测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("GDI-NN 快速测试")
    print("=" * 80)
    
    # 创建测试数据
    create_test_data()
    
    # 运行测试
    results = {}
    
    results['数据加载'] = test_data_loading()
    results['SolvGNN前向传播'] = test_model_forward()
    results['SolvGNN训练步骤'] = test_training_step()
    results['SolvGNNxMLP前向传播'] = test_solvgnn_xmlp_forward()
    results['SolvGNNxMLP训练步骤'] = test_solvgnn_xmlp_training()
    results['GEGNN前向传播'] = test_gegnn_forward()
    results['GEGNN训练步骤'] = test_gegnn_training()
    results['MCM前向传播'] = test_mcm_forward()
    results['MCM训练步骤'] = test_mcm_training()
    results['GibbsDuhemLoss'] = test_gibbs_duhem_loss()
    results['GibbsDuhemLoss与模型集成'] = test_gibbs_duhem_loss_with_model()
    results['预测'] = test_prediction()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print("=" * 80)
    
    if passed == total:
        print("✓ 所有测试通过！")
        print("\n现在可以使用以下命令进行完整训练:")
        print("  python train_gdinn.py \\")
        print("    --model_type SolvGNN \\")
        print("    --batch_size 32 \\")
        print("    --epochs 2 \\")
        print("    --hidden_dim 64 \\")
        print("    --lr 1e-3 \\")
        print("    --pinn_lambda 1.0")
        print("\n训练完成后，可以使用以下命令进行预测:")
        print("  python predict_gdinn.py \\")
        print("    --model_type SolvGNN \\")
        print("    --batch_size 32 \\")
        print("    --hidden_dim 64 \\")
        print("    --checkpoint ./checkpoints/best_model.pdparams")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
