import torch
from xlstm.blocks.mlstm.block_m import mLSTMBlock, mLSTMBlockConfig
from xlstm.blocks.mlstm.layer_m import mLSTMLayerConfig


def test_mlstm_block():
    # Create a configuration for the block
    config = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            embedding_dim=32,
            context_length=16,
            conv1d_kernel_size=4,
            qkv_proj_blocksize=8,
            num_heads=4,
            proj_factor=2.0,
            bias=True,
            dropout=0.1,
        ),
        _num_blocks=1,
        _block_idx=0,
    )

    # Create the mLSTMBlock
    mlstm_block = mLSTMBlock(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Determine the device from the model parameters
    device = next(mlstm_block.parameters()).device

    # Dummy input tensor
    B, S, E = 4, 16, 32  # Batch size, Sequence length, Embedding dimension
    x = torch.randn(B, S, E).to(device)

    # Forward pass
    y = mlstm_block(x)

    # Assertions
    assert y.shape == x.shape, f"Expected output shape {x.shape}, but got {y.shape}"
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    test_mlstm_block()
