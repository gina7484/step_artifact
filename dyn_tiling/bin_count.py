import torch
import numpy as np


batch = 1024  # 256, 512, 1024
layer = 24  # 0 (even), 24 (middle), 30 (uneven)
expert_selection_file = f"/home/ginasohn/expert_routing/processed_qwen/expr_large_b/token1_layer{layer}_b{batch}.npz"
expert_indices_npz = np.load(expert_selection_file)
expert_indices = torch.from_numpy(
    expert_indices_npz["arr_0"]
)  # [B, n_activated_experts]

# expert_counts: [n_routed_experts] (bincount across all batches)
expert_counts = torch.bincount(expert_indices.flatten(), minlength=128)
print(f"Expert counts: {expert_counts}")
