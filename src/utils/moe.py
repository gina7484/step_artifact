import torch
from typing import List, Optional
import torch.nn.functional as F


def topk_to_multihot(
    topk_indices: torch.Tensor,  # [B*N, active_experts]
    num_experts: int,
) -> torch.Tensor:
    """
    vector of indices of selected expert => multihot vector

    (e.g., [1,2,5] => [0,1,1,0,0,1])

    [B*N, active_experts] => [B*N, num_experts]
    """
    # pylint: disable=not-callable
    one_hot_selection = F.one_hot(
        topk_indices, num_classes=num_experts
    )  # [B*N, active_experts, num_experts]
    return one_hot_selection.sum(dim=-2)  # [B*N, num_experts]


def topk_to_onehot(
    topk_indices: torch.Tensor,  # [B*N, active_experts]
    num_experts: int,
) -> torch.Tensor:
    """
    vector of indices of selected expert => one-hot vector

    (e.g., [1,2,5] => [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1]])

    [B*N, active_experts] => [B*N, active_experts, num_experts]
    """
    # pylint: disable=not-callable
    one_hot_selection = F.one_hot(
        topk_indices, num_classes=num_experts
    )  # [B*N, active_experts, num_experts]
    return one_hot_selection  # [B*N, active_experts, num_experts]


def expert_distribution_from_topk(
    topk_indices: torch.Tensor,  # [B*N, active_experts]
    num_experts: int,
) -> List[int]:
    """
    [B*N, active_experts] => [num_experts]
    """
    return torch.bincount(topk_indices.flatten(), minlength=num_experts).tolist()


def moe_gold_calc(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: Optional[torch.Tensor],
    linear_gate_list: List[torch.nn.Linear],
    linear_up_list: List[torch.nn.Linear],
    linear_down_list: List[torch.nn.Linear],
) -> torch.Tensor:
    """
    Args:
        input_tensor: Tensor of shape [B * N, dim]
        expert_indices: Tensor of shape [B* N, active_experts] with integer values [0, num_experts-1]
        expert_weights: Tensor of shape [B* N, active_experts]
        linear_gate_list: List of linear layers for gating
        linear_up_list: List of linear layers for up projection
        linear_down_list: List of linear layers for down projection
    Returns:
        Tensor of shape [B, N, MLP_HID]
    """
    num_experts = len(linear_gate_list)
    y = torch.zeros_like(input_tensor)
    counts = torch.bincount(expert_indices.flatten(), minlength=num_experts).tolist()
    for i in range(num_experts):
        if counts[i] == 0:
            continue
        idx, top = torch.where(expert_indices == i)
        expert_i = linear_down_list[i](
            F.silu(linear_gate_list[i](input_tensor[idx]))
            * linear_up_list[i](input_tensor[idx])
        )  # [B*N, dim]
        if expert_weights is not None:
            y[idx] += expert_i * expert_weights[idx, top, None]
        else:
            y[idx] += expert_i
    return y


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
    shape = input_tensor.size()
    H = shape[-1]  # Last dimension is H
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

    output = result.view(shape[:-1] + (MLP_HID,))  # [B, N, MLP_HID]

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
