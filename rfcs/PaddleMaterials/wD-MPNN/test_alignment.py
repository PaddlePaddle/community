"""
Alignment test: verifies that the PaddleMaterials conversion of polymer-chemprop
produces numerically equivalent results to the original PyTorch implementation.

Tests:
1. Featurization alignment: MolGraph produces identical features
2. BatchMolGraph alignment: batch construction produces identical tensors
3. Model output alignment: given identical weights, outputs match within tolerance

Usage:
    python test_alignment.py

Requirements:
    - Original polymer-chemprop installed (PyTorch)
    - PaddleMaterials polymer_chemprop module (PaddlePaddle)
"""

import sys
import os
import warnings

import numpy as np

# Ensure PaddleMaterials root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Also ensure polymer-chemprop is on the path
POLYMER_CHEMPROP_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "Projects", "github",
    "material_workspace", "polymer-chemprop"
)
# Try alternate paths
if not os.path.exists(POLYMER_CHEMPROP_ROOT):
    POLYMER_CHEMPROP_ROOT = "/home/shun/workspace/Projects/github/paddle_material_workspace/polymer-chemprop"

ATOL = 1e-5  # absolute tolerance for numerical comparison
RTOL = 1e-4  # relative tolerance


def check_torch_available():
    """Check if PyTorch and original chemprop are available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_chemprop_available():
    """Check if original chemprop is available."""
    try:
        sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
        from chemprop.features.featurization import MolGraph as TorchMolGraph  # noqa: F401
        return True
    except ImportError:
        return False


def test_featurization_alignment():
    """Test that atom/bond features are identical between implementations."""
    import paddle
    from ppmat.models.polymer_chemprop.featurization import (
        Featurization_parameters,
        MolGraph as PaddleMolGraph,
        atom_features as paddle_atom_features,
        bond_features as paddle_bond_features,
        onek_encoding_unk as paddle_onek,
    )

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        atom_features as torch_atom_features,
        bond_features as torch_bond_features,
        onek_encoding_unk as torch_onek,
        reset_featurization_parameters,
    )

    # Reset original chemprop to default params
    reset_featurization_parameters()

    # Test onek_encoding
    test_cases = [(5, list(range(10))), (0, [0, 1, 2, 3]), (99, list(range(100)))]
    for val, choices in test_cases:
        p_enc = paddle_onek(val, choices)
        t_enc = torch_onek(val, choices)
        assert p_enc == t_enc, f"onek_encoding mismatch for val={val}: {p_enc} vs {t_enc}"
    print("  onek_encoding_unk: ALIGNED")

    # Test atom features with RDKit molecule
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    for atom in mol.GetAtoms():
        config = Featurization_parameters()
        p_feat = paddle_atom_features(atom, config=config)
        t_feat = torch_atom_features(atom)
        assert p_feat == t_feat, f"atom_features mismatch for atom {atom.GetIdx()}"
    print("  atom_features: ALIGNED")

    # Test bond features
    for bond in mol.GetBonds():
        config = Featurization_parameters()
        p_feat = paddle_bond_features(bond, config=config)
        t_feat = torch_bond_features(bond)
        assert p_feat == t_feat, f"bond_features mismatch for bond {bond.GetIdx()}"
    print("  bond_features: ALIGNED")

    # Test MolGraph construction
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "C1CCCCC1"]
    for smi in smiles_list:
        config = Featurization_parameters()
        p_mg = PaddleMolGraph(smi, config=config)
        t_mg = TorchMolGraph(smi)

        assert p_mg.n_atoms == t_mg.n_atoms, f"n_atoms mismatch for {smi}: {p_mg.n_atoms} vs {t_mg.n_atoms}"
        assert p_mg.n_bonds == t_mg.n_bonds, f"n_bonds mismatch for {smi}: {p_mg.n_bonds} vs {t_mg.n_bonds}"
        assert p_mg.f_atoms == t_mg.f_atoms, f"f_atoms mismatch for {smi}"
        assert p_mg.f_bonds == t_mg.f_bonds, f"f_bonds mismatch for {smi}"
        assert p_mg.a2b == t_mg.a2b, f"a2b mismatch for {smi}"
        assert p_mg.b2a == t_mg.b2a, f"b2a mismatch for {smi}"
        assert p_mg.b2revb == t_mg.b2revb, f"b2revb mismatch for {smi}"
        assert p_mg.w_atoms == t_mg.w_atoms, f"w_atoms mismatch for {smi}"
        assert p_mg.w_bonds == t_mg.w_bonds, f"w_bonds mismatch for {smi}"
    print(f"  MolGraph construction: ALIGNED (tested {len(smiles_list)} molecules)")
    print("  PASS: test_featurization_alignment")


def test_batch_mol_graph_alignment():
    """Test that BatchMolGraph produces identical tensors."""
    import torch
    import paddle
    from ppmat.models.polymer_chemprop.featurization import (
        Featurization_parameters,
        MolGraph as PaddleMolGraph,
        BatchMolGraph as PaddleBatchMolGraph,
    )

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        BatchMolGraph as TorchBatchMolGraph,
        reset_featurization_parameters,
    )
    reset_featurization_parameters()

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]

    config = Featurization_parameters()
    p_graphs = [PaddleMolGraph(s, config=config) for s in smiles_list]
    t_graphs = [TorchMolGraph(s) for s in smiles_list]

    p_batch = PaddleBatchMolGraph(p_graphs)
    t_batch = TorchBatchMolGraph(t_graphs)

    # Compare tensor values
    p_f_atoms = p_batch.f_atoms.numpy()
    t_f_atoms = t_batch.f_atoms.numpy()
    np.testing.assert_allclose(p_f_atoms, t_f_atoms, atol=ATOL, rtol=RTOL,
                               err_msg="f_atoms mismatch")
    print("  f_atoms: ALIGNED")

    p_f_bonds = p_batch.f_bonds.numpy()
    t_f_bonds = t_batch.f_bonds.numpy()
    np.testing.assert_allclose(p_f_bonds, t_f_bonds, atol=ATOL, rtol=RTOL,
                               err_msg="f_bonds mismatch")
    print("  f_bonds: ALIGNED")

    p_w_atoms = p_batch.w_atoms.numpy()
    t_w_atoms = t_batch.w_atoms.numpy()
    np.testing.assert_allclose(p_w_atoms, t_w_atoms, atol=ATOL, rtol=RTOL,
                               err_msg="w_atoms mismatch")
    print("  w_atoms: ALIGNED")

    p_w_bonds = p_batch.w_bonds.numpy()
    t_w_bonds = t_batch.w_bonds.numpy()
    np.testing.assert_allclose(p_w_bonds, t_w_bonds, atol=ATOL, rtol=RTOL,
                               err_msg="w_bonds mismatch")
    print("  w_bonds: ALIGNED")

    p_a2b = p_batch.a2b.numpy()
    t_a2b = t_batch.a2b.numpy()
    np.testing.assert_array_equal(p_a2b, t_a2b, err_msg="a2b mismatch")
    print("  a2b: ALIGNED")

    p_b2a = p_batch.b2a.numpy()
    t_b2a = t_batch.b2a.numpy()
    np.testing.assert_array_equal(p_b2a, t_b2a, err_msg="b2a mismatch")
    print("  b2a: ALIGNED")

    p_b2revb = p_batch.b2revb.numpy()
    t_b2revb = t_batch.b2revb.numpy()
    np.testing.assert_array_equal(p_b2revb, t_b2revb, err_msg="b2revb mismatch")
    print("  b2revb: ALIGNED")

    assert p_batch.a_scope == t_batch.a_scope, "a_scope mismatch"
    assert p_batch.b_scope == t_batch.b_scope, "b_scope mismatch"
    print("  a_scope, b_scope: ALIGNED")

    print("  PASS: test_batch_mol_graph_alignment")


def test_model_weight_transfer_and_output():
    """Test that transferring weights from PyTorch to Paddle produces identical outputs.

    This test:
    1. Creates a PyTorch MoleculeModel with specific args
    2. Creates a PaddlePaddle PolymerChempropModel with matching architecture
    3. Copies weights from PyTorch to Paddle
    4. Runs the same input through both
    5. Compares outputs
    """
    import torch
    import paddle

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        BatchMolGraph as TorchBatchMolGraph,
        get_atom_fdim as torch_get_atom_fdim,
        get_bond_fdim as torch_get_bond_fdim,
        reset_featurization_parameters,
    )
    from chemprop.models.mpn import MPNEncoder as TorchMPNEncoder

    from ppmat.models.polymer_chemprop.featurization import (
        Featurization_parameters,
        MolGraph as PaddleMolGraph,
        BatchMolGraph as PaddleBatchMolGraph,
        get_atom_fdim as paddle_get_atom_fdim,
        get_bond_fdim as paddle_get_bond_fdim,
    )
    from ppmat.models.polymer_chemprop.mpn import MPNEncoder as PaddleMPNEncoder

    reset_featurization_parameters()
    config = Featurization_parameters()

    hidden_size = 64
    depth = 3

    # Verify dimensions match
    t_atom_fdim = torch_get_atom_fdim()
    t_bond_fdim = torch_get_bond_fdim()
    p_atom_fdim = paddle_get_atom_fdim(config)
    p_bond_fdim = paddle_get_bond_fdim(config)

    assert t_atom_fdim == p_atom_fdim, f"atom_fdim mismatch: {t_atom_fdim} vs {p_atom_fdim}"
    assert t_bond_fdim == p_bond_fdim, f"bond_fdim mismatch: {t_bond_fdim} vs {p_bond_fdim}"
    print(f"  Dimensions match: atom_fdim={t_atom_fdim}, bond_fdim={t_bond_fdim}")

    # Create a minimal mock args for torch MPNEncoder
    class MockArgs:
        def __init__(self):
            self.atom_messages = False
            self.hidden_size = hidden_size
            self.bias = False
            self.depth = depth
            self.dropout = 0.0
            self.undirected = False
            self.device = 'cpu'
            self.aggregation = 'mean'
            self.aggregation_norm = 100
            self.activation = 'ReLU'
            self.atom_descriptors = None
            self.atom_descriptors_size = 0

    args = MockArgs()

    # Create both encoders
    torch_encoder = TorchMPNEncoder(args, t_atom_fdim, t_bond_fdim)
    torch_encoder.eval()

    paddle_encoder = PaddleMPNEncoder(
        atom_fdim=p_atom_fdim,
        bond_fdim=p_bond_fdim,
        hidden_size=hidden_size,
        depth=depth,
        dropout=0.0,
        bias=False,
        undirected=False,
        atom_messages=False,
        aggregation='mean',
        aggregation_norm=100,
        activation='ReLU',
    )
    paddle_encoder.eval()

    # Transfer weights from PyTorch to Paddle
    torch_state = torch_encoder.state_dict()
    paddle_state = paddle_encoder.state_dict()

    weight_mapping = {
        'W_i.weight': 'W_i.weight',
        'W_h.weight': 'W_h.weight',
        'W_o.weight': 'W_o.weight',
        'W_o.bias': 'W_o.bias',
    }

    for t_name, p_name in weight_mapping.items():
        if t_name in torch_state and p_name in paddle_state:
            t_val = torch_state[t_name].detach().cpu().numpy()
            # PyTorch Linear weight is [out, in], Paddle is [in, out] -> transpose
            if t_name.endswith('.weight') and t_val.ndim == 2:
                t_val = t_val.T
            paddle_state[p_name] = paddle.to_tensor(t_val)
    
    # Also transfer cached_zero_vector
    if 'cached_zero_vector' in torch_state:
        paddle_state['cached_zero_vector'] = paddle.to_tensor(
            torch_state['cached_zero_vector'].detach().cpu().numpy()
        )

    paddle_encoder.set_state_dict(paddle_state)
    print("  Weights transferred from PyTorch to Paddle")

    # Create identical inputs
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]

    t_graphs = [TorchMolGraph(s) for s in smiles_list]
    t_batch = TorchBatchMolGraph(t_graphs)

    p_graphs = [PaddleMolGraph(s, config=config) for s in smiles_list]
    p_batch = PaddleBatchMolGraph(p_graphs)

    # Run forward pass
    with torch.no_grad():
        t_output = torch_encoder(t_batch).detach().cpu().numpy()

    with paddle.no_grad():
        p_output = paddle_encoder(p_batch).numpy()

    print(f"  PyTorch output shape: {t_output.shape}")
    print(f"  Paddle output shape: {p_output.shape}")
    print(f"  PyTorch output (first 5): {t_output[0, :5]}")
    print(f"  Paddle output (first 5): {p_output[0, :5]}")

    max_diff = np.max(np.abs(t_output - p_output))
    print(f"  Max absolute difference: {max_diff:.2e}")

    np.testing.assert_allclose(p_output, t_output, atol=1e-4, rtol=1e-3,
                               err_msg="MPNEncoder output mismatch")
    print("  MPNEncoder output: ALIGNED")
    print("  PASS: test_model_weight_transfer_and_output")


def test_full_model_alignment():
    """Test full model (MPN + FFN) alignment between PyTorch and Paddle."""
    import torch
    import paddle

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        BatchMolGraph as TorchBatchMolGraph,
        reset_featurization_parameters,
    )
    from chemprop.models.model import MoleculeModel as TorchModel

    from ppmat.models.polymer_chemprop.featurization import (
        Featurization_parameters,
        MolGraph as PaddleMolGraph,
        BatchMolGraph as PaddleBatchMolGraph,
    )
    from ppmat.models.polymer_chemprop.model import PolymerChempropModel as PaddleModel

    reset_featurization_parameters()
    config = Featurization_parameters()

    hidden_size = 64
    depth = 2
    ffn_num_layers = 2
    ffn_hidden_size = 64
    num_tasks = 1

    # Create mock args for PyTorch model
    class MockArgs:
        def __init__(self):
            self.atom_messages = False
            self.hidden_size = hidden_size
            self.bias = False
            self.depth = depth
            self.dropout = 0.0
            self.undirected = False
            self.device = 'cpu'
            self.aggregation = 'mean'
            self.aggregation_norm = 100
            self.activation = 'ReLU'
            self.atom_descriptors = None
            self.atom_descriptors_size = 0
            self.features_only = False
            self.use_input_features = False
            self.features_size = 0
            self.overwrite_default_atom_features = False
            self.overwrite_default_bond_features = False
            self.number_of_molecules = 1
            self.mpn_shared = False
            self.num_tasks = num_tasks
            self.dataset_type = 'regression'
            self.multiclass_num_classes = 3
            self.ffn_num_layers = ffn_num_layers
            self.ffn_hidden_size = ffn_hidden_size
            self.checkpoint_frzn = None
            self.freeze_first_only = False
            self.frzn_ffn_layers = 0
            self.spectra_activation = 'softplus'

    args = MockArgs()
    torch_model = TorchModel(args)
    torch_model.eval()

    paddle_model = PaddleModel(
        hidden_size=hidden_size,
        depth=depth,
        dropout=0.0,
        ffn_num_layers=ffn_num_layers,
        ffn_hidden_size=ffn_hidden_size,
        num_tasks=num_tasks,
        dataset_type='regression',
        property_name='target',
    )
    paddle_model.eval()

    # Transfer all weights
    torch_sd = torch_model.state_dict()
    paddle_sd = paddle_model.state_dict()

    # Build mapping: torch key -> paddle key
    # The model structure is:
    # encoder.encoder[0].{W_i, W_h, W_o, cached_zero_vector}
    # ffn.{0=dropout, 1=linear, 2=act, 3=dropout, 4=linear}
    transferred = 0
    skipped = []
    for t_key, t_val in torch_sd.items():
        p_key = t_key  # same key structure
        if p_key in paddle_sd:
            t_np = t_val.detach().cpu().numpy()
            # PyTorch Linear weight is [out, in], Paddle is [in, out] -> transpose
            if t_key.endswith('.weight') and t_np.ndim == 2:
                t_np = t_np.T
            paddle_sd[p_key] = paddle.to_tensor(t_np)
            transferred += 1
        else:
            skipped.append(t_key)

    paddle_model.set_state_dict(paddle_sd)
    print(f"  Transferred {transferred} parameters, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped keys: {skipped}")

    # Create identical inputs
    smiles_list = ["CCO", "c1ccccc1"]

    t_graphs = [TorchMolGraph(s) for s in smiles_list]
    t_batch = TorchBatchMolGraph(t_graphs)

    p_graphs = [PaddleMolGraph(s, config=config) for s in smiles_list]
    p_batch = PaddleBatchMolGraph(p_graphs)

    # Run forward pass
    with torch.no_grad():
        t_output = torch_model([t_batch]).detach().cpu().numpy()

    data = {
        "batch_graphs": [p_batch],
    }
    with paddle.no_grad():
        result = paddle_model(data, return_loss=False, return_prediction=True)
        p_output = result["pred_dict"]["target"].numpy()

    print(f"  PyTorch output: {t_output.flatten()}")
    print(f"  Paddle output: {p_output.flatten()}")

    max_diff = np.max(np.abs(t_output - p_output))
    print(f"  Max absolute difference: {max_diff:.2e}")

    np.testing.assert_allclose(p_output, t_output, atol=1e-4, rtol=1e-3,
                               err_msg="Full model output mismatch")
    print("  Full model output: ALIGNED")
    print("  PASS: test_full_model_alignment")


def _create_aligned_models():
    """Helper: create PyTorch and Paddle models with identical weights.

    Returns (torch_model, paddle_model, config, MockArgs) with weights transferred.
    """
    import torch
    import paddle

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import reset_featurization_parameters
    from chemprop.models.model import MoleculeModel as TorchModel

    from ppmat.models.polymer_chemprop.featurization import Featurization_parameters
    from ppmat.models.polymer_chemprop.model import PolymerChempropModel as PaddleModel

    reset_featurization_parameters()
    config = Featurization_parameters()

    hidden_size = 64
    depth = 2
    ffn_num_layers = 2
    ffn_hidden_size = 64
    num_tasks = 1

    class MockArgs:
        def __init__(self):
            self.atom_messages = False
            self.hidden_size = hidden_size
            self.bias = False
            self.depth = depth
            self.dropout = 0.0
            self.undirected = False
            self.device = 'cpu'
            self.aggregation = 'mean'
            self.aggregation_norm = 100
            self.activation = 'ReLU'
            self.atom_descriptors = None
            self.atom_descriptors_size = 0
            self.features_only = False
            self.use_input_features = False
            self.features_size = 0
            self.overwrite_default_atom_features = False
            self.overwrite_default_bond_features = False
            self.number_of_molecules = 1
            self.mpn_shared = False
            self.num_tasks = num_tasks
            self.dataset_type = 'regression'
            self.multiclass_num_classes = 3
            self.ffn_num_layers = ffn_num_layers
            self.ffn_hidden_size = ffn_hidden_size
            self.checkpoint_frzn = None
            self.freeze_first_only = False
            self.frzn_ffn_layers = 0
            self.spectra_activation = 'softplus'

    args = MockArgs()
    torch_model = TorchModel(args)

    paddle_model = PaddleModel(
        hidden_size=hidden_size,
        depth=depth,
        dropout=0.0,
        ffn_num_layers=ffn_num_layers,
        ffn_hidden_size=ffn_hidden_size,
        num_tasks=num_tasks,
        dataset_type='regression',
        property_name='target',
    )

    # Transfer weights
    torch_sd = torch_model.state_dict()
    paddle_sd = paddle_model.state_dict()
    for t_key, t_val in torch_sd.items():
        p_key = t_key
        if p_key in paddle_sd:
            t_np = t_val.detach().cpu().numpy()
            if t_key.endswith('.weight') and t_np.ndim == 2:
                t_np = t_np.T
            paddle_sd[p_key] = paddle.to_tensor(t_np)
    paddle_model.set_state_dict(paddle_sd)

    return torch_model, paddle_model, config, args


def _make_batches(smiles_list, config):
    """Helper: create PyTorch and Paddle BatchMolGraph from SMILES list."""
    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        BatchMolGraph as TorchBatchMolGraph,
    )
    from ppmat.models.polymer_chemprop.featurization import (
        MolGraph as PaddleMolGraph,
        BatchMolGraph as PaddleBatchMolGraph,
    )

    t_graphs = [TorchMolGraph(s) for s in smiles_list]
    t_batch = TorchBatchMolGraph(t_graphs)
    p_graphs = [PaddleMolGraph(s, config=config) for s in smiles_list]
    p_batch = PaddleBatchMolGraph(p_graphs)
    return t_batch, p_batch


def test_predict_alignment():
    """Test that the predict interface produces identical outputs between PyTorch and Paddle.

    This uses model.eval() mode and the Paddle model's predict() convenience method.
    """
    import torch
    import paddle

    torch_model, paddle_model, config, _ = _create_aligned_models()
    torch_model.eval()
    paddle_model.eval()

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC"]
    t_batch, p_batch = _make_batches(smiles_list, config)

    # PyTorch predict
    with torch.no_grad():
        t_output = torch_model([t_batch]).detach().cpu().numpy()

    # Paddle predict (via convenience method)
    p_result = paddle_model.predict([p_batch])
    p_output = p_result["target"]

    print(f"  PyTorch predict output: {t_output.flatten()}")
    print(f"  Paddle predict output:  {p_output.flatten()}")

    max_diff = np.max(np.abs(t_output - p_output))
    print(f"  Max absolute difference: {max_diff:.2e}")

    np.testing.assert_allclose(p_output, t_output, atol=1e-4, rtol=1e-3,
                               err_msg="Predict output mismatch")
    print("  Predict output: ALIGNED")
    print("  PASS: test_predict_alignment")


def test_train_alignment():
    """Test that one training step produces aligned loss and gradient updates.

    Given identical initial weights, data, and optimizer config (SGD with same lr),
    verify:
    1. Forward loss is identical
    2. After one optimizer step, updated weights are identical
    3. Forward output after the step is identical
    """
    import torch
    import paddle

    torch_model, paddle_model, config, _ = _create_aligned_models()
    torch_model.train()
    paddle_model.train()

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    t_batch, p_batch = _make_batches(smiles_list, config)

    # Fake regression labels
    np.random.seed(42)
    labels_np = np.random.randn(len(smiles_list), 1).astype(np.float32)

    # --- Step 1: Compare forward loss ---
    t_labels = torch.tensor(labels_np)
    t_output = torch_model([t_batch])
    t_loss = torch.nn.functional.mse_loss(t_output, t_labels)
    t_loss_val = t_loss.item()

    p_labels = paddle.to_tensor(labels_np)
    data = {
        "batch_graphs": [p_batch],
        "labels": p_labels,
    }
    result = paddle_model(data, return_loss=True, return_prediction=True)
    p_loss = result["loss_dict"]["loss"]
    p_loss_val = p_loss.item()
    p_output = result["pred_dict"]["target"].numpy()

    print(f"  PyTorch loss: {t_loss_val:.8f}")
    print(f"  Paddle loss:  {p_loss_val:.8f}")

    np.testing.assert_allclose(p_loss_val, t_loss_val, atol=1e-5, rtol=1e-4,
                               err_msg="Training loss mismatch")
    print("  Forward loss: ALIGNED")

    # --- Step 2: One optimizer step, compare updated weights ---
    lr = 0.01
    t_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
    p_optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=paddle_model.parameters())

    t_optimizer.zero_grad()
    t_loss.backward()
    t_optimizer.step()

    p_loss.backward()
    p_optimizer.step()
    p_optimizer.clear_grad()

    # Compare updated weights
    t_sd = torch_model.state_dict()
    p_sd = paddle_model.state_dict()

    max_weight_diff = 0.0
    for t_key, t_val in t_sd.items():
        if t_key in p_sd:
            t_np = t_val.detach().cpu().numpy()
            p_np = p_sd[t_key].numpy()
            if t_key.endswith('.weight') and t_np.ndim == 2:
                t_np = t_np.T
            diff = np.max(np.abs(t_np - p_np))
            max_weight_diff = max(max_weight_diff, diff)

    print(f"  Max weight difference after 1 step: {max_weight_diff:.2e}")
    assert max_weight_diff < 1e-4, f"Weight difference too large after 1 step: {max_weight_diff:.2e}"
    print("  Updated weights: ALIGNED")

    # --- Step 3: Forward again with updated weights ---
    torch_model.eval()
    paddle_model.eval()

    with torch.no_grad():
        t_output2 = torch_model([t_batch]).detach().cpu().numpy()

    with paddle.no_grad():
        result2 = paddle_model(data, return_loss=False, return_prediction=True)
        p_output2 = result2["pred_dict"]["target"].numpy()

    max_diff2 = np.max(np.abs(t_output2 - p_output2))
    print(f"  Max output difference after 1 step: {max_diff2:.2e}")

    np.testing.assert_allclose(p_output2, t_output2, atol=1e-4, rtol=1e-3,
                               err_msg="Output mismatch after training step")
    print("  Post-step output: ALIGNED")
    print("  PASS: test_train_alignment")


TORCH_TEST_DATA_DIR = os.path.join(POLYMER_CHEMPROP_ROOT, "tests", "data")


def _load_regression_csv(csv_path, max_rows=None):
    """Load SMILES and targets from a regression CSV file.

    Returns (smiles_list, targets) where targets is np.ndarray of shape (N, num_tasks).
    """
    import csv
    smiles_list = []
    targets = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            smiles_list.append(row[0])
            targets.append([float(x) for x in row[1:]])
    return smiles_list, np.array(targets, dtype=np.float32)


def _transfer_weights_torch_to_paddle(torch_model, paddle_model):
    """Transfer all weights from a PyTorch model to a Paddle model, transposing Linear weights."""
    import paddle
    torch_sd = torch_model.state_dict()
    paddle_sd = paddle_model.state_dict()
    for t_key, t_val in torch_sd.items():
        if t_key in paddle_sd:
            t_np = t_val.detach().cpu().numpy()
            if t_key.endswith('.weight') and t_np.ndim == 2:
                t_np = t_np.T
            paddle_sd[t_key] = paddle.to_tensor(t_np)
    paddle_model.set_state_dict(paddle_sd)


def test_real_data_predict_alignment():
    """Test predict alignment using real regression data from tests/data/regression.csv.

    Loads real SMILES molecules from the original project's test data,
    creates models with identical weights, and verifies predictions match.
    """
    import torch
    import paddle

    csv_path = os.path.join(TORCH_TEST_DATA_DIR, "regression.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test data not found: {csv_path}")

    smiles_list, targets = _load_regression_csv(csv_path, max_rows=50)
    print(f"  Loaded {len(smiles_list)} molecules from regression.csv")

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        BatchMolGraph as TorchBatchMolGraph,
        reset_featurization_parameters,
    )
    from chemprop.models.model import MoleculeModel as TorchModel

    from ppmat.models.polymer_chemprop.featurization import (
        Featurization_parameters,
        MolGraph as PaddleMolGraph,
        BatchMolGraph as PaddleBatchMolGraph,
    )
    from ppmat.models.polymer_chemprop.model import PolymerChempropModel as PaddleModel

    reset_featurization_parameters()
    config = Featurization_parameters()

    hidden_size = 64
    depth = 3
    ffn_num_layers = 2
    ffn_hidden_size = 64
    num_tasks = targets.shape[1]

    class MockArgs:
        def __init__(self):
            self.atom_messages = False
            self.hidden_size = hidden_size
            self.bias = False
            self.depth = depth
            self.dropout = 0.0
            self.undirected = False
            self.device = 'cpu'
            self.aggregation = 'mean'
            self.aggregation_norm = 100
            self.activation = 'ReLU'
            self.atom_descriptors = None
            self.atom_descriptors_size = 0
            self.features_only = False
            self.use_input_features = False
            self.features_size = 0
            self.overwrite_default_atom_features = False
            self.overwrite_default_bond_features = False
            self.number_of_molecules = 1
            self.mpn_shared = False
            self.num_tasks = num_tasks
            self.dataset_type = 'regression'
            self.multiclass_num_classes = 3
            self.ffn_num_layers = ffn_num_layers
            self.ffn_hidden_size = ffn_hidden_size
            self.checkpoint_frzn = None
            self.freeze_first_only = False
            self.frzn_ffn_layers = 0
            self.spectra_activation = 'softplus'

    torch_model = TorchModel(MockArgs())
    paddle_model = PaddleModel(
        hidden_size=hidden_size, depth=depth, dropout=0.0,
        ffn_num_layers=ffn_num_layers, ffn_hidden_size=ffn_hidden_size,
        num_tasks=num_tasks, dataset_type='regression', property_name='target',
    )
    _transfer_weights_torch_to_paddle(torch_model, paddle_model)
    torch_model.eval()
    paddle_model.eval()

    # Process in batches (batch_size=10)
    batch_size = 10
    all_t_preds = []
    all_p_preds = []
    for start in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[start:start + batch_size]

        t_graphs = [TorchMolGraph(s) for s in batch_smiles]
        t_batch = TorchBatchMolGraph(t_graphs)
        p_graphs = [PaddleMolGraph(s, config=config) for s in batch_smiles]
        p_batch = PaddleBatchMolGraph(p_graphs)

        with torch.no_grad():
            t_out = torch_model([t_batch]).detach().cpu().numpy()
        p_result = paddle_model.predict([p_batch])
        p_out = p_result["target"]

        all_t_preds.append(t_out)
        all_p_preds.append(p_out)

    t_preds = np.concatenate(all_t_preds, axis=0)
    p_preds = np.concatenate(all_p_preds, axis=0)

    max_diff = np.max(np.abs(t_preds - p_preds))
    print(f"  Total predictions: {t_preds.shape[0]}")
    print(f"  Max absolute difference: {max_diff:.2e}")

    np.testing.assert_allclose(p_preds, t_preds, atol=1e-4, rtol=1e-3,
                               err_msg="Real data predict mismatch")
    print("  Real data predict: ALIGNED")
    print("  PASS: test_real_data_predict_alignment")


def test_real_data_train_alignment():
    """Test training alignment using real regression data from tests/data/regression.csv.

    Loads real molecules, trains both PyTorch and Paddle models for multiple epochs
    with identical initial weights and SGD optimizer, and verifies:
    1. Per-batch loss matches at each step
    2. Final predictions on a held-out set match
    """
    import torch
    import paddle

    csv_path = os.path.join(TORCH_TEST_DATA_DIR, "regression.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test data not found: {csv_path}")

    smiles_list, targets = _load_regression_csv(csv_path, max_rows=100)

    # Split: first 80 for training, last 20 for testing
    train_smiles = smiles_list[:80]
    train_targets = targets[:80]
    test_smiles = smiles_list[80:]

    # Normalize targets (like chemprop does for regression)
    target_mean = train_targets.mean(axis=0)
    target_std = train_targets.std(axis=0)
    target_std = np.where(target_std == 0, 1.0, target_std)
    train_targets_norm = (train_targets - target_mean) / target_std

    print(f"  Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print(f"  Target mean: {target_mean}, std: {target_std}")

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import (
        MolGraph as TorchMolGraph,
        BatchMolGraph as TorchBatchMolGraph,
        reset_featurization_parameters,
    )
    from chemprop.models.model import MoleculeModel as TorchModel

    from ppmat.models.polymer_chemprop.featurization import (
        Featurization_parameters,
        MolGraph as PaddleMolGraph,
        BatchMolGraph as PaddleBatchMolGraph,
    )
    from ppmat.models.polymer_chemprop.model import PolymerChempropModel as PaddleModel

    reset_featurization_parameters()
    config = Featurization_parameters()

    hidden_size = 64
    depth = 3
    ffn_num_layers = 2
    ffn_hidden_size = 64
    num_tasks = targets.shape[1]

    class MockArgs:
        def __init__(self):
            self.atom_messages = False
            self.hidden_size = hidden_size
            self.bias = False
            self.depth = depth
            self.dropout = 0.0
            self.undirected = False
            self.device = 'cpu'
            self.aggregation = 'mean'
            self.aggregation_norm = 100
            self.activation = 'ReLU'
            self.atom_descriptors = None
            self.atom_descriptors_size = 0
            self.features_only = False
            self.use_input_features = False
            self.features_size = 0
            self.overwrite_default_atom_features = False
            self.overwrite_default_bond_features = False
            self.number_of_molecules = 1
            self.mpn_shared = False
            self.num_tasks = num_tasks
            self.dataset_type = 'regression'
            self.multiclass_num_classes = 3
            self.ffn_num_layers = ffn_num_layers
            self.ffn_hidden_size = ffn_hidden_size
            self.checkpoint_frzn = None
            self.freeze_first_only = False
            self.frzn_ffn_layers = 0
            self.spectra_activation = 'softplus'

    torch.manual_seed(42)
    torch_model = TorchModel(MockArgs())

    paddle_model = PaddleModel(
        hidden_size=hidden_size, depth=depth, dropout=0.0,
        ffn_num_layers=ffn_num_layers, ffn_hidden_size=ffn_hidden_size,
        num_tasks=num_tasks, dataset_type='regression', property_name='target',
    )
    _transfer_weights_torch_to_paddle(torch_model, paddle_model)

    # Optimizers
    lr = 1e-3
    t_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
    p_optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=paddle_model.parameters())

    # Training loop
    batch_size = 20
    num_epochs = 3
    max_loss_diff = 0.0

    for epoch in range(num_epochs):
        torch_model.train()
        paddle_model.train()

        for start in range(0, len(train_smiles), batch_size):
            end = min(start + batch_size, len(train_smiles))
            batch_smiles = train_smiles[start:end]
            batch_targets = train_targets_norm[start:end]

            # Build graphs
            t_graphs = [TorchMolGraph(s) for s in batch_smiles]
            t_batch = TorchBatchMolGraph(t_graphs)
            p_graphs = [PaddleMolGraph(s, config=config) for s in batch_smiles]
            p_batch = PaddleBatchMolGraph(p_graphs)

            # PyTorch forward + backward
            t_labels = torch.tensor(batch_targets)
            t_optimizer.zero_grad()
            t_preds = torch_model([t_batch])
            t_loss = torch.nn.functional.mse_loss(t_preds, t_labels)
            t_loss.backward()
            t_optimizer.step()

            # Paddle forward + backward
            p_labels = paddle.to_tensor(batch_targets)
            data = {"batch_graphs": [p_batch], "labels": p_labels}
            result = paddle_model(data, return_loss=True, return_prediction=False)
            p_loss = result["loss_dict"]["loss"]
            p_loss.backward()
            p_optimizer.step()
            p_optimizer.clear_grad()

            loss_diff = abs(t_loss.item() - p_loss.item())
            max_loss_diff = max(max_loss_diff, loss_diff)

        print(f"  Epoch {epoch}: max batch loss diff so far = {max_loss_diff:.2e}")

    print(f"  Max loss difference across all batches: {max_loss_diff:.2e}")
    assert max_loss_diff < 1e-4, f"Training loss diverged: max diff = {max_loss_diff:.2e}"
    print("  Per-batch training loss: ALIGNED")

    # Compare predictions on test set after training
    torch_model.eval()
    paddle_model.eval()

    t_test_graphs = [TorchMolGraph(s) for s in test_smiles]
    t_test_batch = TorchBatchMolGraph(t_test_graphs)
    p_test_graphs = [PaddleMolGraph(s, config=config) for s in test_smiles]
    p_test_batch = PaddleBatchMolGraph(p_test_graphs)

    with torch.no_grad():
        t_test_out = torch_model([t_test_batch]).detach().cpu().numpy()
    # Inverse scale
    t_test_preds = t_test_out * target_std + target_mean

    p_result = paddle_model.predict([p_test_batch])
    p_test_out = p_result["target"]
    p_test_preds = p_test_out * target_std + target_mean

    max_pred_diff = np.max(np.abs(t_test_preds - p_test_preds))
    print(f"  Post-training test predictions max diff: {max_pred_diff:.2e}")

    np.testing.assert_allclose(p_test_preds, t_test_preds, atol=1e-3, rtol=1e-2,
                               err_msg="Post-training prediction mismatch on real data")
    print("  Post-training predictions: ALIGNED")
    print("  PASS: test_real_data_train_alignment")


def _make_torch_model_and_args(hidden_size=64, depth=3, ffn_num_layers=2,
                               ffn_hidden_size=64, num_tasks=1):
    """Helper: create a PyTorch MoleculeModel and MockArgs."""
    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.models.model import MoleculeModel as TorchModel

    class MockArgs:
        def __init__(self):
            self.atom_messages = False
            self.hidden_size = hidden_size
            self.bias = False
            self.depth = depth
            self.dropout = 0.0
            self.undirected = False
            self.device = 'cpu'
            self.aggregation = 'mean'
            self.aggregation_norm = 100
            self.activation = 'ReLU'
            self.atom_descriptors = None
            self.atom_descriptors_size = 0
            self.features_only = False
            self.use_input_features = False
            self.features_size = 0
            self.overwrite_default_atom_features = False
            self.overwrite_default_bond_features = False
            self.number_of_molecules = 1
            self.mpn_shared = False
            self.num_tasks = num_tasks
            self.dataset_type = 'regression'
            self.multiclass_num_classes = 3
            self.ffn_num_layers = ffn_num_layers
            self.ffn_hidden_size = ffn_hidden_size
            self.checkpoint_frzn = None
            self.freeze_first_only = False
            self.frzn_ffn_layers = 0
            self.spectra_activation = 'softplus'

    args = MockArgs()
    return TorchModel(args), args


def _make_paddle_model(hidden_size=64, depth=3, ffn_num_layers=2,
                       ffn_hidden_size=64, num_tasks=1):
    """Helper: create a Paddle PolymerChempropModel."""
    from ppmat.models.polymer_chemprop.model import PolymerChempropModel as PaddleModel
    return PaddleModel(
        hidden_size=hidden_size, depth=depth, dropout=0.0,
        ffn_num_layers=ffn_num_layers, ffn_hidden_size=ffn_hidden_size,
        num_tasks=num_tasks, dataset_type='regression', property_name='target',
    )


def test_pipeline_predict_alignment():
    """Test predict alignment using the full PaddleMaterials pipeline.

    Paddle side: PolymerChempropDataset → PolymerChempropCollator → Model.predict
    PyTorch side: MoleculeDatapoint → construct_molecule_batch → Model forward

    Uses real regression.csv data from the original project's tests/data/.
    """
    import torch
    import paddle

    csv_path = os.path.join(TORCH_TEST_DATA_DIR, "regression.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test data not found: {csv_path}")

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import reset_featurization_parameters
    from chemprop.data.data import MoleculeDatapoint, MoleculeDataset, construct_molecule_batch

    from ppmat.datasets.polymer_chemprop_dataset import PolymerChempropDataset
    from ppmat.datasets.collate_fn import PolymerChempropCollator

    reset_featurization_parameters()

    num_tasks = 1
    torch_model, _ = _make_torch_model_and_args(num_tasks=num_tasks)
    paddle_model = _make_paddle_model(num_tasks=num_tasks)
    _transfer_weights_torch_to_paddle(torch_model, paddle_model)
    torch_model.eval()
    paddle_model.eval()

    # --- Paddle side: full pipeline ---
    paddle_dataset = PolymerChempropDataset(
        path=csv_path,
        smiles_columns=['smiles'],
        target_columns=['logSolubility'],
        max_data_size=50,
    )
    collator = PolymerChempropCollator()
    print(f"  Paddle dataset: {len(paddle_dataset)} samples")

    # Process in batches via collator
    batch_size = 10
    all_p_preds = []
    for start in range(0, len(paddle_dataset), batch_size):
        end = min(start + batch_size, len(paddle_dataset))
        samples = [paddle_dataset[i] for i in range(start, end)]
        batch = collator(samples)
        p_result = paddle_model.predict(batch["batch_graphs"])
        all_p_preds.append(p_result["target"])
    p_preds = np.concatenate(all_p_preds, axis=0)

    # --- PyTorch side: use MoleculeDatapoint + construct_molecule_batch ---
    smiles_list, targets = _load_regression_csv(csv_path, max_rows=50)
    all_t_preds = []
    for start in range(0, len(smiles_list), batch_size):
        end = min(start + batch_size, len(smiles_list))
        batch_smiles = smiles_list[start:end]
        batch_targets = targets[start:end].tolist()

        data_points = [
            MoleculeDatapoint(smiles=[s], targets=t)
            for s, t in zip(batch_smiles, batch_targets)
        ]
        torch_batch = construct_molecule_batch(data_points)
        mol_batch = torch_batch.batch_graph()

        with torch.no_grad():
            t_out = torch_model(mol_batch).detach().cpu().numpy()
        all_t_preds.append(t_out)
    t_preds = np.concatenate(all_t_preds, axis=0)

    max_diff = np.max(np.abs(t_preds - p_preds))
    print(f"  Total predictions: {p_preds.shape[0]}")
    print(f"  Max absolute difference: {max_diff:.2e}")

    np.testing.assert_allclose(p_preds, t_preds, atol=1e-4, rtol=1e-3,
                               err_msg="Pipeline predict mismatch")
    print("  Pipeline predict: ALIGNED")
    print("  PASS: test_pipeline_predict_alignment")


def test_pipeline_train_alignment():
    """Test training alignment using the full PaddleMaterials pipeline.

    Paddle side: PolymerChempropDataset → PolymerChempropCollator → Model → loss → backward
    PyTorch side: MoleculeDatapoint → construct_molecule_batch → Model → loss → backward

    Verifies:
    1. Per-batch loss matches at each step
    2. Final predictions on held-out set match after training
    """
    import torch
    import paddle

    csv_path = os.path.join(TORCH_TEST_DATA_DIR, "regression.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test data not found: {csv_path}")

    sys.path.insert(0, POLYMER_CHEMPROP_ROOT)
    from chemprop.features.featurization import reset_featurization_parameters
    from chemprop.data.data import MoleculeDatapoint, MoleculeDataset, construct_molecule_batch

    from ppmat.datasets.polymer_chemprop_dataset import PolymerChempropDataset
    from ppmat.datasets.collate_fn import PolymerChempropCollator

    reset_featurization_parameters()

    num_tasks = 1
    torch.manual_seed(42)
    torch_model, _ = _make_torch_model_and_args(num_tasks=num_tasks)
    paddle_model = _make_paddle_model(num_tasks=num_tasks)
    _transfer_weights_torch_to_paddle(torch_model, paddle_model)

    # Load data
    smiles_list, targets_all = _load_regression_csv(csv_path, max_rows=100)
    train_size = 80

    # Normalize targets (regression)
    train_targets = targets_all[:train_size]
    target_mean = train_targets.mean(axis=0)
    target_std = train_targets.std(axis=0)
    target_std = np.where(target_std == 0, 1.0, target_std)

    # --- Paddle side: build dataset from CSV, then normalize targets in-place ---
    # Write a temp CSV with normalized targets for the paddle dataset
    import tempfile, csv as csv_mod
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow(['smiles', 'target'])
        for i in range(train_size):
            norm_val = (targets_all[i, 0] - target_mean[0]) / target_std[0]
            writer.writerow([smiles_list[i], f'{norm_val:.8f}'])
        train_csv_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow(['smiles', 'target'])
        for i in range(train_size, len(smiles_list)):
            writer.writerow([smiles_list[i], f'{targets_all[i, 0]:.8f}'])
        test_csv_path = f.name

    try:
        paddle_train_dataset = PolymerChempropDataset(
            path=train_csv_path,
            smiles_columns=['smiles'],
            target_columns=['target'],
        )
        paddle_test_dataset = PolymerChempropDataset(
            path=test_csv_path,
            smiles_columns=['smiles'],
            target_columns=['target'],
        )
        collator = PolymerChempropCollator()

        # Normalized targets for PyTorch side
        train_targets_norm = (targets_all[:train_size] - target_mean) / target_std

        # Optimizers
        lr = 1e-3
        t_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
        p_optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=paddle_model.parameters())

        batch_size = 20
        num_epochs = 3
        max_loss_diff = 0.0

        for epoch in range(num_epochs):
            torch_model.train()
            paddle_model.train()

            for start in range(0, train_size, batch_size):
                end = min(start + batch_size, train_size)

                # Paddle side: dataset → collator
                p_samples = [paddle_train_dataset[i] for i in range(start, end)]
                p_batch = collator(p_samples)
                result = paddle_model(p_batch, return_loss=True, return_prediction=False)
                p_loss = result["loss_dict"]["loss"]
                p_loss.backward()
                p_optimizer.step()
                p_optimizer.clear_grad()

                # PyTorch side: MoleculeDatapoint → construct_molecule_batch
                batch_smiles = smiles_list[start:end]
                batch_targets_norm = train_targets_norm[start:end].tolist()
                data_points = [
                    MoleculeDatapoint(smiles=[s], targets=t)
                    for s, t in zip(batch_smiles, batch_targets_norm)
                ]
                torch_batch_ds = construct_molecule_batch(data_points)
                mol_batch = torch_batch_ds.batch_graph()
                t_targets_list = torch_batch_ds.targets()
                t_labels = torch.tensor([[0 if x is None else x for x in tb] for tb in t_targets_list])

                t_optimizer.zero_grad()
                t_preds = torch_model(mol_batch)
                t_loss = torch.nn.functional.mse_loss(t_preds, t_labels)
                t_loss.backward()
                t_optimizer.step()

                loss_diff = abs(t_loss.item() - p_loss.item())
                max_loss_diff = max(max_loss_diff, loss_diff)

            print(f"  Epoch {epoch}: max batch loss diff so far = {max_loss_diff:.2e}")

        print(f"  Max loss difference across all batches: {max_loss_diff:.2e}")
        assert max_loss_diff < 1e-4, f"Pipeline training loss diverged: max diff = {max_loss_diff:.2e}"
        print("  Per-batch training loss: ALIGNED")

        # --- Compare predictions on test set after training ---
        torch_model.eval()
        paddle_model.eval()

        # Paddle side: test dataset → collator → predict
        p_test_samples = [paddle_test_dataset[i] for i in range(len(paddle_test_dataset))]
        p_test_batch = collator(p_test_samples)
        p_result = paddle_model.predict(p_test_batch["batch_graphs"])
        p_test_out = p_result["target"]
        # Inverse scale
        p_test_preds = p_test_out * target_std + target_mean

        # PyTorch side
        test_smiles = smiles_list[train_size:]
        test_targets = targets_all[train_size:].tolist()
        data_points = [
            MoleculeDatapoint(smiles=[s], targets=t)
            for s, t in zip(test_smiles, test_targets)
        ]
        torch_test_ds = construct_molecule_batch(data_points)
        mol_batch = torch_test_ds.batch_graph()
        with torch.no_grad():
            t_test_out = torch_model(mol_batch).detach().cpu().numpy()
        t_test_preds = t_test_out * target_std + target_mean

        max_pred_diff = np.max(np.abs(t_test_preds - p_test_preds))
        print(f"  Post-training test predictions max diff: {max_pred_diff:.2e}")

        np.testing.assert_allclose(p_test_preds, t_test_preds, atol=1e-3, rtol=1e-2,
                                   err_msg="Pipeline post-training prediction mismatch")
        print("  Post-training predictions: ALIGNED")
        print("  PASS: test_pipeline_train_alignment")
    finally:
        os.unlink(train_csv_path)
        os.unlink(test_csv_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Alignment Test: polymer-chemprop PyTorch vs PaddlePaddle")
    print("=" * 60)

    torch_available = check_torch_available()
    chemprop_available = check_chemprop_available()

    if not torch_available:
        print("\nWARNING: PyTorch not available. Skipping alignment tests.")
        print("Install PyTorch to run alignment tests:")
        print("  pip install torch")
        sys.exit(0)

    if not chemprop_available:
        print(f"\nWARNING: Original chemprop not found at {POLYMER_CHEMPROP_ROOT}")
        print("Skipping alignment tests.")
        sys.exit(0)

    print(f"\nUsing polymer-chemprop from: {POLYMER_CHEMPROP_ROOT}")

    tests = [
        ("Featurization Alignment", test_featurization_alignment),
        ("BatchMolGraph Alignment", test_batch_mol_graph_alignment),
        ("MPNEncoder Weight Transfer & Output", test_model_weight_transfer_and_output),
        ("Full Model Alignment", test_full_model_alignment),
        ("Predict Alignment", test_predict_alignment),
        ("Train Alignment", test_train_alignment),
        ("Real Data Predict Alignment", test_real_data_predict_alignment),
        ("Real Data Train Alignment", test_real_data_train_alignment),
        ("Pipeline Predict Alignment", test_pipeline_predict_alignment),
        ("Pipeline Train Alignment", test_pipeline_train_alignment),
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
