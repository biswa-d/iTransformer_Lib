import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from xlstm.blocks.mlstm.layer_m import mLSTMLayer, mLSTMLayerConfig


def test_mlstm_layer_forward():
    # Create a configuration for the layer
    config = mLSTMLayerConfig(
        embedding_dim=32,
        context_length=16,
        conv1d_kernel_size=4,
        qkv_proj_blocksize=8,
        num_heads=4,
        proj_factor=2.0,
        bias=True,
        dropout=0.1,
    )
    # Create the mLSTM Layer
    mlstm_layer = mLSTMLayer(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input tensor
    B, S, E = 4, 16, 32  # Batch size, Sequence length, Embedding dimension
    x = torch.randn(B, S, E).to(mlstm_layer.proj_up.weight.device)

    # Forward pass
    y = mlstm_layer(x)

    # Assertions
    assert y.shape == x.shape, f"Expected output shape {x.shape}, but got {y.shape}"
    print(f"Output shape (forward): {y.shape}")


def test_mlstm_layer_step():
    # Create a configuration for the layer
    config = mLSTMLayerConfig(
        embedding_dim=32,
        context_length=16,
        conv1d_kernel_size=4,
        qkv_proj_blocksize=8,
        num_heads=4,
        proj_factor=2.0,
        bias=True,
        dropout=0.1,
    )
    # Create the mLSTM Layer
    mlstm_layer = mLSTMLayer(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input tensor
    B, S, E = 4, 1, 32  # Batch size, Sequence length (1 for step), Embedding dimension
    x = torch.randn(B, S, E).to(mlstm_layer.proj_up.weight.device)

    # Initial states
    mlstm_state = None  # No initial state
    conv_state = None  # No initial convolutional state

    # Step pass
    y, state = mlstm_layer.step(x, mlstm_state=mlstm_state, conv_state=conv_state)

    # Assertions
    assert y.shape == (B, S, E), f"Expected output shape {(B, S, E)}, but got {y.shape}"
    assert "mlstm_state" in state, "'mlstm_state' missing in returned state"
    assert "conv_state" in state, "'conv_state' missing in returned state"

    # Verify state shapes
    mlstm_state_shapes = [s.shape for s in state["mlstm_state"]]
    conv_state_shapes = [s.shape for s in state["conv_state"]]

    print(f"Output shape (step): {y.shape}")
    print(f"mLSTM state shapes: {mlstm_state_shapes}")
    print(f"Conv state shapes: {conv_state_shapes}")


if __name__ == "__main__":
    test_mlstm_layer_forward()
    test_mlstm_layer_step()
