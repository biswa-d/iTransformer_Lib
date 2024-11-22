import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from cell_m import mLSTMCell, mLSTMCellConfig  # Import the refactored cell

def test_mlstm_cell_forward():
    # Define configuration
    config = mLSTMCellConfig(
        context_length=16,  # Sequence length
        embedding_dim=32,   # Embedding dimension
        num_heads=8         # Number of heads
    )
    
    # Initialize mLSTMCell
    mlstm_cell = mLSTMCell(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Define inputs
    B, S, H = 4, config.context_length, config.embedding_dim  # Batch size, Sequence length, Hidden size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(B, S, H, device=device)
    k = torch.randn(B, S, H, device=device)
    v = torch.randn(B, S, H, device=device)

    # Run forward pass
    output = mlstm_cell(q, k, v)
    print("Output shape (forward):", output.shape)

    # Assert expected output shape
    assert output.shape == (B, S, H), f"Unexpected output shape: {output.shape}"

def test_mlstm_cell_step():
    # Define configuration
    config = mLSTMCellConfig(
        context_length=1,  # Step operates on one timestep at a time
        embedding_dim=32,  # Embedding dimension
        num_heads=8        # Number of heads
    )
    
    # Initialize mLSTMCell
    mlstm_cell = mLSTMCell(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Define inputs
    B, S, H = 4, 1, config.embedding_dim  # Batch size, Sequence length (S=1 for step), Hidden size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(B, S, H, device=device)
    k = torch.randn(B, S, H, device=device)
    v = torch.randn(B, S, H, device=device)

    # Initial state
    c_state = torch.zeros(B, config.num_heads, H // config.num_heads, H // config.num_heads, device=device)
    n_state = torch.zeros(B, config.num_heads, H // config.num_heads, 1, device=device)
    m_state = torch.zeros(B, config.num_heads, 1, 1, device=device)
    initial_state = (c_state, n_state, m_state)

    # Run step
    output, new_state = mlstm_cell.step(q, k, v, initial_state)
    print("Output shape (step):", output.shape)
    print("New state shapes:", [state.shape for state in new_state])

    # Assert expected output and state shapes
    assert output.shape == (B, S, H), f"Unexpected output shape: {output.shape}"
    assert new_state[0].shape == c_state.shape, "Unexpected c_state shape"
    assert new_state[1].shape == n_state.shape, "Unexpected n_state shape"
    assert new_state[2].shape == m_state.shape, "Unexpected m_state shape"

if __name__ == "__main__":
    test_mlstm_cell_forward()
    test_mlstm_cell_step()
