# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import math
from typing import Optional
import torch


def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: Optional[torch.Tensor] = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """Parallel mLSTM cell with GPU optimization and stabilization."""
    B, NH, S, DH = queries.shape
    device = queries.device

    # 1. Forget gate matrix (log-sigmoid for stability)
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    ltr = (
        lower_triangular_matrix
        if lower_triangular_matrix is not None
        else torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    )

    # 2. Log-forget gates cumulative sum for decay matrix
    log_fgates_cumsum = torch.cat(
        [torch.zeros((B, NH, 1, 1), device=device), torch.cumsum(log_fgates, dim=-2)],
        dim=-2,
    )  # (B, NH, S+1, 1)
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    log_fg_matrix = torch.where(ltr, log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # 3. Gate decay matrix D
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    max_log_D = torch.max(log_D_matrix, dim=-1, keepdim=True)[0]  # Stabilize row-wise
    log_D_matrix_stabilized = log_D_matrix - max_log_D
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    # 4. Scaled keys
    keys_scaled = keys / math.sqrt(DH)

    # 5. Combination matrix C
    qk_matrix = torch.matmul(queries, keys_scaled.transpose(-2, -1))  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.clamp(C_matrix.sum(dim=-1, keepdim=True), min=eps)  # Stabilize with epsilon
    C_matrix_normalized = C_matrix / normalizer  # Normalize

    # 6. Retrieve values
    h_tilde_state = torch.matmul(C_matrix_normalized, values)  # (B, NH, S, DH)
    return h_tilde_state


def recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Single mLSTM step optimized for GPU."""
    B, NH, _, DH = q.shape
    device = q.device

    # Projections
    q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)  # (B, NH, DH)
    q, k, v = q.unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)  # (B, NH, DH, 1)

    # Gates
    log_fg_act = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, 1, 1)
    m_state_new = torch.maximum(log_fg_act + m_state, igate_preact)  # Update m_state
    fg_act = torch.exp(log_fg_act + m_state - m_state_new)
    ig_act = torch.exp(igate_preact - m_state_new)

    # Updated states
    k_scaled = k / math.sqrt(DH)
    c_state_new = fg_act * c_state + ig_act * torch.matmul(k_scaled, v.transpose(-1, -2))  # (B, NH, DH, DH)
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    # Hidden state computation
    h_num = torch.matmul(q.transpose(-1, -2), c_state_new)  # (B, NH, 1, DH)
    qn_dotproduct = torch.matmul(q.transpose(-1, -2), n_state_new)  # (B, NH, 1, 1)
    h_denom = torch.clamp(qn_dotproduct.abs(), min=torch.exp(-m_state_new)) + eps  # Stability
    h = h_num / h_denom  # (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)


def chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    initial_C: Optional[torch.Tensor] = None,
    initial_n: Optional[torch.Tensor] = None,
    initial_m: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Chunkwise processing optimized for GPU."""
    B, NH, S, DH = queries.shape
    device = queries.device

    # Split into chunks
    NS, CS = S // chunk_size, chunk_size
    q_chunks = queries.view(B, NH, NS, CS, DH) / math.sqrt(DH)
    k_chunks = keys.view(B, NH, NS, CS, DH)
    v_chunks = values.view(B, NH, NS, CS, DH)

    # Initialize states
    C = torch.zeros((B, NH, NS + 1, DH, DH), device=device)
    n = torch.zeros((B, NH, NS + 1, DH), device=device)
    m = torch.zeros((B, NH, NS + 1, 1, 1), device=device)

    if initial_C is not None:
        C[:, :, 0] = initial_C
    if initial_n is not None:
        n[:, :, 0] = initial_n
    if initial_m is not None:
        m[:, :, 0] = initial_m

    # Chunkwise processing
    for i in range(NS):
        m[:, :, i + 1] = torch.maximum(
            torch.cumsum(fgate_preact[:, :, i, :], dim=-1).sum(dim=-1, keepdim=True), m[:, :, i]
        )
        C[:, :, i + 1] = (
            C[:, :, i] * torch.exp(m[:, :, i] - m[:, :, i + 1]) + torch.matmul(k_chunks[:, :, i], v_chunks[:, :, i])
        )
        n[:, :, i + 1] = (
            n[:, :, i] * torch.exp(m[:, :, i] - m[:, :, i + 1]) + k_chunks[:, :, i].sum(dim=-2)
        )

    # Combine intra- and inter-chunk results
    intra = torch.matmul(q_chunks, v_chunks.transpose(-1, -2))  # (B, NH, NS, CS, DH)
    inter = torch.matmul(q_chunks, C[:, :, :-1])  # Inter-chunk contributions
    results = intra + inter

    if return_last_state:
        return results.view(B, NH, S, DH), (C[:, :, -1], n[:, :, -1], m[:, :, -1])
    return results.view(B, NH, S, DH)
