from dataclasses import dataclass
import numpy as np
import torch
import csv
import time

from dyn_tiling.test_weight_stationary_gemm_revet_sec3_figure import (
    run_ws_tile_mn_mk_revet_figure,
)

# from step_py.ops import *
# from step_py.functions import map_accum_fn, map_fn, init_fn, accum_fn
# from utils.gold_checking import check_gold_tensor
# from utils.draw_graph import save_graph_format
# from rewrite.broadcast import infer_broadcast
# from utils.moe import *


@dataclass
class SmallerMixtral:  # 8x scaled down version for each dimension
    n_routed_experts = 8
    n_activated_experts = 2
    dim = 512  # 4096/8
    moe_inter_dim = 1792  # 14336/8 (Can use tile size upto 256)


@dataclass
class TinyMixtral:  # 64x scaled down version for each dimension
    n_routed_experts = 8
    n_activated_experts = 2
    dim = 64  # 4096/64
    moe_inter_dim = 224  # 14336/64 (Can use tile size upto 32)


@dataclass
class Mixtral8x7b:
    n_routed_experts = 8
    n_activated_experts = 2
    dim = 4096
    moe_inter_dim = 14336  # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json#L11


def test_gemm_sweep():
    mock_bf16 = True
    # ------------ Model Configuration ------------
    # model_config = SmallerMixtral()
    # model_config = TinyMixtral()
    model_config = Mixtral8x7b()

    tile_Ns = [16]  # For the batch dim (64)
    tile_Fs = [64]  # For the model_config.moe_inter_dim

    # ------------ Expert Indices ------------
    iter = 8
    layer = 10
    expert_selection_file = f"/home/ginasohn/expert_routing/processed_mixtral/expr_per_layer/iter_{iter:03d}_layer_{layer:03d}.npz"
    expert_indices_npz = np.load(expert_selection_file)
    expert_indices = torch.from_numpy(
        expert_indices_npz["data"]
    )  # [B, n_activated_experts]

    # expert_counts: [n_routed_experts] (bincount across all batches)
    expert_counts = torch.bincount(
        expert_indices.flatten(), minlength=model_config.n_routed_experts
    )
    print(f"Expert counts: {expert_counts}")

    # ------------ Input generation -----------
    B = expert_indices.shape[0]

    # Set the random seed
    seed = 5
    torch.manual_seed(seed)

    input_tensor = torch.randn(B, model_config.dim)

    for tile_N in tile_Ns:
        for tile_F in tile_Fs:
            results = []

            off_chip_traffic, on_chip_requirement, cycles, duration_s, _ = (
                run_ws_tile_mn_mk_revet_figure(
                    tile_N,
                    tile_F,
                    input_tensor,
                    expert_indices,
                    model_config,
                    None,
                    False,
                    mock_bf16,
                    # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
                )
            )

            # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
            # num_tiles = [
            #     (routed_toks + tile_N - 1) // tile_N
            #     for routed_toks in expert_counts.tolist()
            # ]
            # after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

            # padded_rows = [
            #     total_toks - raw_toks
            #     for total_toks, raw_toks in zip(
            #         after_pad_batch_dim, expert_counts.tolist()
            #     )
            # ]

            # flops = sum(
            #     [
            #         (
            #             2 * b * model_config.dim * model_config.moe_inter_dim * 3
            #         )  # 3 (Linear layers)
            #         + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            #         + (
            #             8 * b * model_config.dim * model_config.moe_inter_dim
            #         )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            #         for b in after_pad_batch_dim
            #     ]
            # )

            # padded_flops = sum(
            #     [
            #         (
            #             2 * b * model_config.dim * model_config.moe_inter_dim * 3
            #         )  # 3 (Linear layers)
            #         + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            #         + (
            #             8 * b * model_config.dim * model_config.moe_inter_dim
            #         )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            #         for b in padded_rows
            #     ]
            # )

            # free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

            # sub_dict = {
            #     symbol: value
            #     for symbol, value in zip(free_symbols, expert_counts.tolist())
            # }

            # off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

            # dict_to_append = {
            #     "batch": B,
            #     "tile_N": tile_N,
            #     "tile_F": tile_F,
            #     "flops": flops,
            #     "padded_flops": padded_flops,
            #     "cycles": cycles,
            #     "duration_s": duration_s,
            #     "off_chip_traffic_bytes": off_chip_traffic_val,
            #     "on_chip_requirement_bytes": on_chip_requirement,
            # }
            # print(dict_to_append)
            # results.append(dict_to_append)

            # out_file = (
            #     f"/home/ginasohn/step_tl/dyn_tiling/mixtral_{model_config.dim}_"
            #     + f"{model_config.moe_inter_dim}_iter{iter:03d}_layer_{layer:03d}_"
            #     + f"n{tile_N}_f{tile_F}_{time.strftime("%d%H%M%S")}.csv"
            # )
            # try:
            #     with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            #         fieldnames = [
            #             "batch",
            #             "tile_N",
            #             "tile_F",
            #             "flops",
            #             "padded_flops",
            #             "cycles",
            #             "duration_s",
            #             "off_chip_traffic_bytes",
            #             "on_chip_requirement_bytes",
            #         ]
            #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #         writer.writeheader()

            #         # Write data rows
            #         for result in results:
            #             writer.writerow(result)

            #     print(f"Results written to {out_file}")
            # except Exception as e:
            #     print(f"Error writing CSV file: {e}")
