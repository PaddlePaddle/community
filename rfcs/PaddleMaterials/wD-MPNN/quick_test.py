"""
Quick test for the polymer-chemprop → PaddleMaterials conversion.

Tests that:
1. Featurization_parameters, MolGraph, BatchMolGraph work correctly
2. MPNEncoder and MPN produce correct output shapes
3. PolymerChempropModel forward pass works end-to-end
4. PolymerChempropDataset loads data correctly
5. PolymerChempropCollator batches data correctly
6. Loss computation works
7. Backward pass (gradient flow) works
"""

import sys
import os
import tempfile
import csv

import numpy as np
import paddle

# Ensure PaddleMaterials root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppmat.models.polymer_chemprop.featurization import (
    Featurization_parameters,
    MolGraph,
    BatchMolGraph,
    mol2graph,
    get_atom_fdim,
    get_bond_fdim,
    atom_features as af_func,
    bond_features as bf_func,
)
from ppmat.models.polymer_chemprop.nn_utils import (
    index_select_ND,
    get_activation_function,
    initialize_weights,
)
from ppmat.models.polymer_chemprop.mpn import MPNEncoder, MPN
from ppmat.models.polymer_chemprop.model import PolymerChempropModel


def test_featurization_config():
    """Test Featurization_parameters creation and dimension computation."""
    config = Featurization_parameters()
    assert config.max_atomic_num == 100
    assert config.ATOM_FDIM > 0
    assert config.BOND_FDIM == 14

    atom_fdim = get_atom_fdim(config)
    bond_fdim = get_bond_fdim(config)
    assert atom_fdim == config.ATOM_FDIM
    assert bond_fdim == config.BOND_FDIM + config.ATOM_FDIM
    print(f"  atom_fdim={atom_fdim}, bond_fdim={bond_fdim}")
    print("  PASS: test_featurization_config")


def test_mol_graph():
    """Test MolGraph construction for a simple molecule."""
    config = Featurization_parameters()
    smiles = "CCO"  # ethanol
    mg = MolGraph(smiles, config=config)

    assert mg.n_atoms > 0, f"Expected n_atoms > 0, got {mg.n_atoms}"
    assert mg.n_bonds > 0, f"Expected n_bonds > 0, got {mg.n_bonds}"
    assert len(mg.f_atoms) == mg.n_atoms
    assert len(mg.f_bonds) == mg.n_bonds
    assert len(mg.a2b) == mg.n_atoms
    assert len(mg.b2a) == mg.n_bonds
    assert len(mg.b2revb) == mg.n_bonds
    print(f"  CCO: n_atoms={mg.n_atoms}, n_bonds={mg.n_bonds}")
    print("  PASS: test_mol_graph")


def test_batch_mol_graph():
    """Test BatchMolGraph construction from multiple MolGraphs."""
    config = Featurization_parameters()
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    mol_graphs = [MolGraph(s, config=config) for s in smiles_list]
    batch = BatchMolGraph(mol_graphs)

    assert batch.f_atoms.shape[0] > 3  # at least 1 padding + atoms
    assert batch.f_bonds.shape[0] > 1
    assert len(batch.a_scope) == 3
    assert len(batch.b_scope) == 3
    assert batch.f_atoms.dtype == paddle.float32
    assert batch.a2b.dtype == paddle.int64

    # Test get_components
    comps = batch.get_components()
    assert len(comps) == 10
    print(f"  Batch: f_atoms shape={batch.f_atoms.shape}, a_scope={batch.a_scope}")
    print("  PASS: test_batch_mol_graph")


def test_mol2graph():
    """Test mol2graph convenience function."""
    config = Featurization_parameters()
    smiles_list = ["CCO", "c1ccccc1"]
    batch = mol2graph(smiles_list, config=config)
    assert isinstance(batch, BatchMolGraph)
    assert len(batch.a_scope) == 2
    print("  PASS: test_mol2graph")


def test_index_select_nd():
    """Test index_select_ND utility."""
    source = paddle.randn([10, 5])
    index = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype='int64')
    result = index_select_ND(source, index)
    assert result.shape == [2, 3, 5], f"Expected [2, 3, 5], got {result.shape}"
    print("  PASS: test_index_select_nd")


def test_activation_functions():
    """Test all supported activation functions."""
    for name in ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']:
        act = get_activation_function(name)
        x = paddle.randn([2, 3])
        y = act(x)
        assert y.shape == [2, 3], f"Failed for {name}"
    print("  PASS: test_activation_functions")


def test_mpn_encoder():
    """Test MPNEncoder forward pass."""
    config = Featurization_parameters()
    atom_fdim = get_atom_fdim(config)
    bond_fdim = get_bond_fdim(config)
    hidden_size = 64

    encoder = MPNEncoder(
        atom_fdim=atom_fdim,
        bond_fdim=bond_fdim,
        hidden_size=hidden_size,
        depth=3,
        dropout=0.0,
    )

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    mol_graphs = [MolGraph(s, config=config) for s in smiles_list]
    batch = BatchMolGraph(mol_graphs)

    output = encoder(batch)
    assert output.shape == [3, hidden_size], f"Expected [3, {hidden_size}], got {output.shape}"
    print(f"  MPNEncoder output shape: {output.shape}")
    print("  PASS: test_mpn_encoder")


def test_mpn():
    """Test MPN wrapper forward pass."""
    config = Featurization_parameters()
    hidden_size = 64

    mpn = MPN(
        hidden_size=hidden_size,
        depth=3,
        dropout=0.0,
        featurization_config=config,
    )

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    mol_graphs = [MolGraph(s, config=config) for s in smiles_list]
    batch = BatchMolGraph(mol_graphs)

    output = mpn([batch])  # List of BatchMolGraph, one per molecule slot
    assert output.shape == [3, hidden_size], f"Expected [3, {hidden_size}], got {output.shape}"
    print(f"  MPN output shape: {output.shape}")
    print("  PASS: test_mpn")


def test_model_forward():
    """Test PolymerChempropModel forward pass with loss."""
    config = Featurization_parameters()
    hidden_size = 64
    num_tasks = 2

    model = PolymerChempropModel(
        hidden_size=hidden_size,
        depth=3,
        dropout=0.0,
        ffn_num_layers=2,
        ffn_hidden_size=64,
        num_tasks=num_tasks,
        dataset_type='regression',
        property_name='target',
    )

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    mol_graphs = [MolGraph(s, config=config) for s in smiles_list]
    batch_graph = BatchMolGraph(mol_graphs)

    data = {
        "batch_graphs": [batch_graph],
        "labels": paddle.randn([3, num_tasks]),
        "label_mask": paddle.ones([3, num_tasks]),
        "features": None,
        "atom_descriptors_batch": None,
    }

    result = model(data, return_loss=True, return_prediction=True)

    assert "loss_dict" in result
    assert "pred_dict" in result
    assert "loss" in result["loss_dict"]
    assert "target" in result["pred_dict"]
    assert result["pred_dict"]["target"].shape == [3, num_tasks]

    loss = result["loss_dict"]["loss"]
    assert not paddle.isnan(loss).item(), "Loss is NaN!"
    print(f"  Model loss: {loss.item():.6f}")
    print(f"  Prediction shape: {result['pred_dict']['target'].shape}")
    print("  PASS: test_model_forward")


def test_model_backward():
    """Test that gradients flow through the model."""
    config = Featurization_parameters()
    model = PolymerChempropModel(
        hidden_size=64,
        depth=2,
        dropout=0.0,
        ffn_num_layers=1,
        ffn_hidden_size=64,
        num_tasks=1,
        dataset_type='regression',
        property_name='target',
    )

    smiles_list = ["CCO", "c1ccccc1"]
    mol_graphs = [MolGraph(s, config=config) for s in smiles_list]
    batch_graph = BatchMolGraph(mol_graphs)

    data = {
        "batch_graphs": [batch_graph],
        "labels": paddle.randn([2, 1]),
        "label_mask": paddle.ones([2, 1]),
    }

    result = model(data, return_loss=True, return_prediction=False)
    loss = result["loss_dict"]["loss"]
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None and paddle.any(p.grad != 0).item():
            has_grad = True
            break
    assert has_grad, "No gradients found in model parameters!"
    print("  PASS: test_model_backward")


def test_dataset_and_collator():
    """Test PolymerChempropDataset and PolymerChempropCollator."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'target'])
        writer.writerow(['CCO', '1.5'])
        writer.writerow(['c1ccccc1', '-0.3'])
        writer.writerow(['CC(=O)O', '0.8'])
        writer.writerow(['CCCC', '2.1'])
        tmp_path = f.name

    try:
        from ppmat.datasets.polymer_chemprop_dataset import PolymerChempropDataset
        from ppmat.datasets.collate_fn import PolymerChempropCollator

        dataset = PolymerChempropDataset(
            path=tmp_path,
            smiles_columns=['smiles'],
            target_columns=['target'],
        )
        assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"

        sample = dataset[0]
        assert "mol_graphs" in sample
        assert "targets" in sample
        assert "target_mask" in sample
        assert len(sample["mol_graphs"]) == 1
        assert isinstance(sample["mol_graphs"][0], MolGraph)
        assert sample["targets"].shape == (1,)
        assert sample["target_mask"][0] == 1.0
        print(f"  Dataset: {len(dataset)} samples, first target = {sample['targets'][0]}")

        # Test collator
        collator = PolymerChempropCollator()
        batch = collator([dataset[i] for i in range(3)])
        assert "batch_graphs" in batch
        assert "labels" in batch
        assert "label_mask" in batch
        assert len(batch["batch_graphs"]) == 1
        assert isinstance(batch["batch_graphs"][0], BatchMolGraph)
        assert batch["labels"].shape == [3, 1]
        print(f"  Collator: labels shape = {batch['labels'].shape}")
        print("  PASS: test_dataset_and_collator")
    finally:
        os.unlink(tmp_path)


def test_end_to_end():
    """End-to-end test: dataset → collator → model → loss → backward."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'value'])
        writer.writerow(['CCO', '1.5'])
        writer.writerow(['c1ccccc1', '-0.3'])
        writer.writerow(['CC(=O)O', '0.8'])
        tmp_path = f.name

    try:
        from ppmat.datasets.polymer_chemprop_dataset import PolymerChempropDataset
        from ppmat.datasets.collate_fn import PolymerChempropCollator

        dataset = PolymerChempropDataset(
            path=tmp_path,
            smiles_columns=['smiles'],
            target_columns=['value'],
        )
        collator = PolymerChempropCollator()
        batch = collator([dataset[i] for i in range(3)])

        model = PolymerChempropModel(
            hidden_size=64,
            depth=2,
            dropout=0.0,
            ffn_num_layers=2,
            ffn_hidden_size=64,
            num_tasks=1,
            dataset_type='regression',
            property_name='value',
        )

        result = model(batch, return_loss=True, return_prediction=True)
        loss = result["loss_dict"]["loss"]
        loss.backward()

        assert not paddle.isnan(loss).item()
        print(f"  End-to-end loss: {loss.item():.6f}")
        print(f"  Predictions: {result['pred_dict']['value'].numpy().flatten()}")
        print("  PASS: test_end_to_end")
    finally:
        os.unlink(tmp_path)


def test_classification_mode():
    """Test classification mode of the model."""
    config = Featurization_parameters()
    model = PolymerChempropModel(
        hidden_size=64,
        depth=2,
        dropout=0.0,
        ffn_num_layers=1,
        ffn_hidden_size=64,
        num_tasks=1,
        dataset_type='classification',
        property_name='active',
    )

    smiles_list = ["CCO", "c1ccccc1"]
    mol_graphs = [MolGraph(s, config=config) for s in smiles_list]
    batch_graph = BatchMolGraph(mol_graphs)

    data = {
        "batch_graphs": [batch_graph],
        "labels": paddle.to_tensor([[1.0], [0.0]]),
        "label_mask": paddle.ones([2, 1]),
    }

    # During training mode: logits returned
    model.train()
    result = model(data, return_loss=True, return_prediction=True)
    assert "loss" in result["loss_dict"]
    pred = result["pred_dict"]["active"]
    # In train mode, classification returns sigmoid probabilities
    assert paddle.all(pred >= 0).item() and paddle.all(pred <= 1).item()
    print(f"  Classification loss: {result['loss_dict']['loss'].item():.6f}")
    print("  PASS: test_classification_mode")


if __name__ == "__main__":
    print("=" * 60)
    print("Quick Test: polymer-chemprop → PaddleMaterials conversion")
    print("=" * 60)

    tests = [
        ("Featurization Config", test_featurization_config),
        ("MolGraph", test_mol_graph),
        ("BatchMolGraph", test_batch_mol_graph),
        ("mol2graph", test_mol2graph),
        ("index_select_ND", test_index_select_nd),
        ("Activation Functions", test_activation_functions),
        ("MPNEncoder", test_mpn_encoder),
        ("MPN", test_mpn),
        ("Model Forward", test_model_forward),
        ("Model Backward", test_model_backward),
        ("Dataset & Collator", test_dataset_and_collator),
        ("End-to-End", test_end_to_end),
        ("Classification Mode", test_classification_mode),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\n[TEST] {name}")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
