import torch


def moe_linear_gold_calc_batched(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_tensors: list[torch.Tensor],
):
    """
    Args:
        input_tensor: Tensor of shape [B, N, H]
        expert_indices: Tensor of shape [B, N] with integer values [0, EXPERT-1]
        expert_tensors: List of EXPERT tensors, each of shape [H, MLP_HID]

    Returns:
        Tensor of shape [B, N, MLP_HID]
    """
    B, N, H = input_tensor.shape
    MLP_HID = expert_tensors[0].shape[1]
    EXPERT = len(expert_tensors)

    # Stack all expert tensors into a single tensor [EXPERT, H, MLP_HID]
    stacked_experts = torch.stack(expert_tensors, dim=0)

    # Flatten the input and indices for easier indexing
    flat_input = input_tensor.view(-1, H)  # [B*N, H]
    flat_indices = expert_indices.view(-1)  # [B*N]

    # Use advanced indexing to select the right expert for each position
    selected_experts = stacked_experts[flat_indices]  # [B*N, H, MLP_HID]

    # Batch matrix multiplication: [B*N, 1, H] @ [B*N, H, MLP_HID] = [B*N, 1, MLP_HID]
    flat_input_expanded = flat_input.unsqueeze(1)  # [B*N, 1, H]
    result = torch.bmm(flat_input_expanded, selected_experts)  # [B*N, 1, MLP_HID]

    # Remove the extra dimension and reshape back
    result = result.squeeze(1)  # [B*N, MLP_HID]
    output = result.view(B, N, MLP_HID)  # [B, N, MLP_HID]

    return output


def test_moe_gold_calc_batched():
    B, N, H = 2, 3, 4
    MLP_HID = 5
    EXPERT = N

    input_tensor = torch.randn(B, N, H)
    expert_indices = torch.tensor([[0, 1, 2], [0, 1, 2]])  # [B,N]

    expert_tensors = [
        torch.randn(H, MLP_HID) for _ in range(EXPERT)
    ]  # [H, MLP_HID] x EXPERT

    # Gold calculation
    input_tensor_unsqueezed = input_tensor.unsqueeze(2)  # [B, N, 1, H]
    expert_tensors_stacked = (
        torch.stack(expert_tensors, dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    )  # [B, EXPERT, H, MLP_HID]
    gold = torch.bmm(
        input_tensor_unsqueezed.view(B * N, 1, H),
        expert_tensors_stacked.reshape(B * N, H, MLP_HID),
    )  # [B * N, 1, MLP_HID]

    gold_formatted = gold.view(B, N, MLP_HID)  # [B, N, MLP_HID]

    output = moe_linear_gold_calc_batched(input_tensor, expert_indices, expert_tensors)
    assert output.shape == (B, N, MLP_HID), "Output shape mismatch"
    assert gold_formatted.shape == (B, N, MLP_HID), "Output shape mismatch"
    assert torch.allclose(output, gold_formatted, atol=1e-6), "Output values mismatch"

    print("Test passed!")
