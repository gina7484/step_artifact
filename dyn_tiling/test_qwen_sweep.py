from dataclasses import dataclass
import numpy as np
import torch
import csv
import time

from dyn_tiling.test_weight_stationary_gemm import run_ws_tile_mn_mk
from dyn_tiling.test_weight_stationary_gemm_revet import run_ws_tile_mn_mk_revet
from dyn_tiling.test_weight_stationary_gemm_dyn_tile import run_ws_tile_mn_mk_dyn_tile

# from step_py.ops import *
# from step_py.functions import map_accum_fn, map_fn, init_fn, accum_fn
# from utils.gold_checking import check_gold_tensor
# from utils.draw_graph import save_graph_format
# from rewrite.broadcast import infer_broadcast
# from utils.moe import *


@dataclass
class SmallerQwen30b:  # 16x scaled down version for each dimension
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 128  # 2048 // 16
    moe_inter_dim = 48  # 768 // 16


@dataclass
class Qwen30b:
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 2048  # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json#L12
    moe_inter_dim = (
        768  # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json#L19
    )


@dataclass
class TinyQwen30b:  # 32x scaled down version for each dimension
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 64  # 2048 // 32
    moe_inter_dim = 24  # 768 // 32


def test_gemm_dyn_tile():
    mock_bf16 = True
    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    # model_config = TinyQwen30b()
    model_config = Qwen30b()

    # tile_Ns = [64]  # For the batch dim (64)
    round_N = 1
    tile_Fs = [64]  # For the model_config.moe_inter_dim

    # ------------ Expert Indices ------------
    batch = 64
    iter = 32
    layer = 12
    expert_selection_file = f"./dyn_tiling/expert_routing/qwen_b{batch}/iter_{iter:03d}_layer_{layer:03d}.npz"
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

    for tile_F in tile_Fs:
        # ------------ Output file ------------
        out_file = (
            f"./dyn_tiling/"
            + f"qwen_{model_config.dim}_{model_config.moe_inter_dim}_"
            + f"round_{round_N}_iter{iter:03d}_layer_{layer:03d}_"
            + f"n_dyn_f{tile_F}_{time.strftime("%d%H%M%S")}.csv"
        )
        # out_file = None

        results = []

        (
            off_chip_traffic,
            on_chip_requirement,
            cycles,
            duration_s,
            unit_expert_on_chip,
        ) = run_ws_tile_mn_mk_dyn_tile(
            round_N,
            tile_F,
            input_tensor,
            expert_indices,
            model_config,
            "timing",  # "full",
            False,
            mock_bf16,
            # logging=f"expert_par_gemm_dyn_tile_round_{round_N}_f{tile_F}",
        )

        # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
        num_tiles = [
            (routed_toks + round_N - 1) // round_N
            for routed_toks in expert_counts.tolist()
        ]
        after_pad_batch_dim = [num_tiles_i * round_N for num_tiles_i in num_tiles]

        padded_rows = [
            total_toks - raw_toks
            for total_toks, raw_toks in zip(after_pad_batch_dim, expert_counts.tolist())
        ]

        flops = sum(
            [
                (
                    2 * b * model_config.dim * model_config.moe_inter_dim * 3
                )  # 3 (Linear layers)
                + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
                + (
                    8 * b * model_config.dim * model_config.moe_inter_dim
                )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                for b in after_pad_batch_dim
            ]
        )

        padded_flops = sum(
            [
                (
                    2 * b * model_config.dim * model_config.moe_inter_dim * 3
                )  # 3 (Linear layers)
                + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
                + (
                    8 * b * model_config.dim * model_config.moe_inter_dim
                )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                for b in padded_rows
            ]
        )
        # --------------- off-chip traffic ---------------
        free_symbols = sorted(off_chip_traffic.free_symbols, key=str)
        # print(f"off_chip_traffic free_symbols: {free_symbols}")

        sub_dict = {
            symbol: value for symbol, value in zip(free_symbols, expert_counts.tolist())
        }
        # print(f"off_chip_traffic: {off_chip_traffic}")
        # print(f"off_chip_traffic sub_dict: {sub_dict}")

        off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

        # --------------- On-chip requirement ---------------
        free_symbols = sorted(on_chip_requirement.free_symbols, key=str)
        # print(f"on_chip_traffic free_symbols: {free_symbols}")

        sub_dict = {
            symbol: value for symbol, value in zip(free_symbols, expert_counts.tolist())
        }
        # print(f"on_chip_traffic: {on_chip_requirement}")
        # print(f"on_chip_traffic sub_dict: {sub_dict}")

        on_chip_requirement_val = on_chip_requirement.subs(sub_dict)
        print(f"on_chip_requirement_val: {on_chip_requirement_val}")
        on_chip_requirement_val = (
            on_chip_requirement_val
            - expert_counts.tolist().count(0) * unit_expert_on_chip
        )
        # subtract the on-chip requirement for unselected experts weight load assuming on-chip memory is allocated on-demand

        # print(f"on_chip_requirement_val after subtract: {on_chip_requirement_val}")

        dict_to_append = {
            "batch": B,
            "round_N": round_N,
            "tile_F": tile_F,
            "flops": flops,
            "padded_flops": padded_flops,
            "cycles": cycles,
            "duration_s": duration_s,
            "off_chip_traffic_bytes": off_chip_traffic_val,
            "on_chip_requirement_bytes": on_chip_requirement_val,
        }
        print(dict_to_append)
        results.append(dict_to_append)

        if out_file is not None:
            try:
                with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "batch",
                        "round_N",
                        "tile_F",
                        "flops",
                        "padded_flops",
                        "cycles",
                        "duration_s",
                        "off_chip_traffic_bytes",
                        "on_chip_requirement_bytes",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()

                    # Write data rows
                    for result in results:
                        writer.writerow(result)

                print(f"Results written to {out_file}")
            except Exception as e:
                print(f"Error writing CSV file: {e}")


def test_gemm_revet_sweep():
    mock_bf16 = True
    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    # model_config = TinyQwen30b()
    model_config = Qwen30b()

    tile_Ns = [4]  # For the batch dim (64)
    tile_Fs = [64]  # For the model_config.moe_inter_dim

    # ------------ Expert Indices ------------
    batch = 64
    iter = 32
    layer = 12
    expert_selection_file = f"./dyn_tiling/expert_routing/qwen_b{batch}/iter_{iter:03d}_layer_{layer:03d}.npz"
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
            # ------------ Output file ------------
            out_file = (
                f"./dyn_tiling/"
                + f"qwen_{model_config.dim}_{model_config.moe_inter_dim}_"
                + f"iter{iter:03d}_layer_{layer:03d}_n{tile_N}_f{tile_F}_revet_"
                + f"{time.strftime("%d%H%M%S")}.csv"
            )
            # out_file = None

            results = []

            (
                off_chip_traffic,
                on_chip_requirement,
                cycles,
                duration_s,
                unit_expert_on_chip,
            ) = run_ws_tile_mn_mk_revet(
                tile_N,
                tile_F,
                input_tensor,
                expert_indices,
                model_config,
                "timing",
                False,
                mock_bf16,
                # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
            )

            # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
            num_tiles = [
                (routed_toks + tile_N - 1) // tile_N
                for routed_toks in expert_counts.tolist()
            ]
            after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

            padded_rows = [
                total_toks - raw_toks
                for total_toks, raw_toks in zip(
                    after_pad_batch_dim, expert_counts.tolist()
                )
            ]

            flops = sum(
                [
                    (
                        2 * b * model_config.dim * model_config.moe_inter_dim * 3
                    )  # 3 (Linear layers)
                    + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
                    + (
                        8 * b * model_config.dim * model_config.moe_inter_dim
                    )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                    for b in after_pad_batch_dim
                ]
            )

            padded_flops = sum(
                [
                    (
                        2 * b * model_config.dim * model_config.moe_inter_dim * 3
                    )  # 3 (Linear layers)
                    + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
                    + (
                        8 * b * model_config.dim * model_config.moe_inter_dim
                    )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                    for b in padded_rows
                ]
            )

            # --------------- off-chip traffic ---------------
            free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

            sub_dict = {
                symbol: value
                for symbol, value in zip(free_symbols, expert_counts.tolist())
            }

            off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

            # --------------- On-chip requirement ---------------
            on_chip_requirement_val = (
                on_chip_requirement
                - expert_counts.tolist().count(0) * unit_expert_on_chip
            )
            # subtract the on-chip requirement for unselected experts

            dict_to_append = {
                "batch": B,
                "tile_N": tile_N,
                "tile_F": tile_F,
                "flops": flops,
                "padded_flops": padded_flops,
                "cycles": cycles,
                "duration_s": duration_s,
                "off_chip_traffic_bytes": off_chip_traffic_val,
                "on_chip_requirement_bytes": on_chip_requirement_val,
            }
            print(dict_to_append)
            results.append(dict_to_append)

            if out_file is not None:
                try:
                    with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
                        fieldnames = [
                            "batch",
                            "tile_N",
                            "tile_F",
                            "flops",
                            "padded_flops",
                            "cycles",
                            "duration_s",
                            "off_chip_traffic_bytes",
                            "on_chip_requirement_bytes",
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        writer.writeheader()

                        # Write data rows
                        for result in results:
                            writer.writerow(result)

                    print(f"Results written to {out_file}")
                except Exception as e:
                    print(f"Error writing CSV file: {e}")


def test_qwen_b64():
    mock_bf16 = True

    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    model_config = Qwen30b()

    # ------------ Expert Indices ------------
    batch = 64
    iter = 32
    layer = 12
    expert_selection_file = f"./dyn_tiling/expert_routing/qwen_b{batch}/iter_{iter:03d}_layer_{layer:03d}.npz"
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
    tile_F = 64 if isinstance(model_config, Qwen30b) else 24

    B = expert_indices.shape[0]

    # Set the random seed
    seed = 5
    torch.manual_seed(seed)

    input_tensor = torch.randn(B, model_config.dim)

    
    # ------------ Result Dict ------------
    result_dict = {
        "cycles": {
            "tile=16": 0,
            "tile=64": 0,
            "tile=dynamic": 0,
        },
        "off_chip_traffic": {
            "tile=16": 0,
            "tile=64": 0,
            "tile=dynamic": 0,
        },
        "on_chip_mem": {
            "tile=16": 0,
            "tile=64": 0,
            "tile=dynamic": 0,
        },
    }

    result_dict_raw = {
        "cycles": {
            "tile=16": 0,
            "tile=64": 0,
            "tile=dynamic": 0,
        },
        "off_chip_traffic": {
            "tile=16": 0,
            "tile=64": 0,
            "tile=dynamic": 0,
        },
        "on_chip_mem": {
            "tile=16": 0,
            "tile=64": 0,
            "tile=dynamic": 0,
        },
    }

    
    ################## Dynamic Tile Size ##################
    round_N = 1

    dyn_cycles = 0
    dyn_off_chip_traffic = 0
    dyn_on_chip_mem = 0
    
    (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        unit_expert_on_chip,
    ) = run_ws_tile_mn_mk_dyn_tile(
        round_N,
        tile_F,
        input_tensor,
        expert_indices,
        model_config,
        "timing",  # "full",
        False,
        mock_bf16,
        # logging=f"expert_par_gemm_dyn_tile_round_{round_N}_f{tile_F}",
    )

    # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
    num_tiles = [
        (routed_toks + round_N - 1) // round_N
        for routed_toks in expert_counts.tolist()
    ]
    after_pad_batch_dim = [num_tiles_i * round_N for num_tiles_i in num_tiles]

    padded_rows = [
        total_toks - raw_toks
        for total_toks, raw_toks in zip(after_pad_batch_dim, expert_counts.tolist())
    ]

    flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in after_pad_batch_dim
        ]
    )

    padded_flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in padded_rows
        ]
    )
    # --------------- off-chip traffic ---------------
    free_symbols = sorted(off_chip_traffic.free_symbols, key=str)
    # print(f"off_chip_traffic free_symbols: {free_symbols}")

    sub_dict = {
        symbol: value for symbol, value in zip(free_symbols, expert_counts.tolist())
    }
    # print(f"off_chip_traffic: {off_chip_traffic}")
    # print(f"off_chip_traffic sub_dict: {sub_dict}")

    off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

    # --------------- On-chip requirement ---------------
    free_symbols = sorted(on_chip_requirement.free_symbols, key=str)
    # print(f"on_chip_traffic free_symbols: {free_symbols}")

    sub_dict = {
        symbol: value for symbol, value in zip(free_symbols, expert_counts.tolist())
    }
    # print(f"on_chip_traffic: {on_chip_requirement}")
    # print(f"on_chip_traffic sub_dict: {sub_dict}")

    on_chip_requirement_val = on_chip_requirement.subs(sub_dict)
    print(f"on_chip_requirement_val: {on_chip_requirement_val}")
    on_chip_requirement_val = (
        on_chip_requirement_val
        - expert_counts.tolist().count(0) * unit_expert_on_chip
    )
    # subtract the on-chip requirement for unselected experts weight load assuming on-chip memory is allocated on-demand

    dyn_cycles = int(cycles)
    dyn_off_chip_traffic = int(off_chip_traffic_val)
    dyn_on_chip_mem = int(on_chip_requirement_val)

    result_dict_raw["cycles"]["tile=dynamic"] = dyn_cycles
    result_dict_raw["off_chip_traffic"]["tile=dynamic"] = dyn_off_chip_traffic
    result_dict_raw["on_chip_mem"]["tile=dynamic"] = dyn_on_chip_mem

    result_dict["cycles"]["tile=dynamic"] = dyn_cycles / dyn_cycles
    result_dict["off_chip_traffic"]["tile=dynamic"] = dyn_off_chip_traffic / dyn_off_chip_traffic
    result_dict["on_chip_mem"]["tile=dynamic"] = dyn_on_chip_mem / dyn_on_chip_mem


    ################## Static Tile Size (16) ##################
    tile_N = 16
    
    (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        unit_expert_on_chip,
    ) = run_ws_tile_mn_mk_revet(
        tile_N,
        tile_F,
        input_tensor,
        expert_indices,
        model_config,
        "timing",
        False,
        mock_bf16,
        # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
    )

    # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
    num_tiles = [
        (routed_toks + tile_N - 1) // tile_N
        for routed_toks in expert_counts.tolist()
    ]
    after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

    padded_rows = [
        total_toks - raw_toks
        for total_toks, raw_toks in zip(
            after_pad_batch_dim, expert_counts.tolist()
        )
    ]

    flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in after_pad_batch_dim
        ]
    )

    padded_flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in padded_rows
        ]
    )

    # --------------- off-chip traffic ---------------
    free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

    sub_dict = {
        symbol: value
        for symbol, value in zip(free_symbols, expert_counts.tolist())
    }

    off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

    # --------------- On-chip requirement ---------------
    on_chip_requirement_val = (
        on_chip_requirement
        - expert_counts.tolist().count(0) * unit_expert_on_chip
    )
    # subtract the on-chip requirement for unselected experts

    cycles = int(cycles)
    off_chip_traffic_val = int(off_chip_traffic_val)
    on_chip_requirement_val = int(on_chip_requirement_val)

    result_dict_raw["cycles"]["tile=16"] = cycles
    result_dict_raw["off_chip_traffic"]["tile=16"] = off_chip_traffic_val
    result_dict_raw["on_chip_mem"]["tile=16"] = on_chip_requirement_val

    result_dict["cycles"]["tile=16"] = cycles / dyn_cycles
    result_dict["off_chip_traffic"]["tile=16"] = off_chip_traffic_val / dyn_off_chip_traffic
    result_dict["on_chip_mem"]["tile=16"] = on_chip_requirement_val / dyn_on_chip_mem

    ################## Static Tile Size (64) ##################
    tile_N = 64

    (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        unit_expert_on_chip,
    ) = run_ws_tile_mn_mk_revet(
        tile_N,
        tile_F,
        input_tensor,
        expert_indices,
        model_config,
        "timing",
        False,
        mock_bf16,
        # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
    )

    # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
    num_tiles = [
        (routed_toks + tile_N - 1) // tile_N
        for routed_toks in expert_counts.tolist()
    ]
    after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

    padded_rows = [
        total_toks - raw_toks
        for total_toks, raw_toks in zip(
            after_pad_batch_dim, expert_counts.tolist()
        )
    ]

    flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in after_pad_batch_dim
        ]
    )

    padded_flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in padded_rows
        ]
    )

    # --------------- off-chip traffic ---------------
    free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

    sub_dict = {
        symbol: value
        for symbol, value in zip(free_symbols, expert_counts.tolist())
    }

    off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

    # --------------- On-chip requirement ---------------
    on_chip_requirement_val = (
        on_chip_requirement
        - expert_counts.tolist().count(0) * unit_expert_on_chip
    )
    # subtract the on-chip requirement for unselected experts


    cycles = int(cycles)
    off_chip_traffic_val = int(off_chip_traffic_val)
    on_chip_requirement_val = int(on_chip_requirement_val)

    result_dict_raw["cycles"]["tile=64"] = cycles
    result_dict_raw["off_chip_traffic"]["tile=64"] = off_chip_traffic_val
    result_dict_raw["on_chip_mem"]["tile=64"] = on_chip_requirement_val

    result_dict["cycles"]["tile=64"] = cycles / dyn_cycles
    result_dict["off_chip_traffic"]["tile=64"] = off_chip_traffic_val / dyn_off_chip_traffic
    result_dict["on_chip_mem"]["tile=64"] = on_chip_requirement_val / dyn_on_chip_mem

    
    ################## Save Results to CSV ##################

    out_file = f"./dyn_tiling/figure_6_qwen_b64.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "metric",
                "tile=16",
                "tile=64",
                "tile=dynamic",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for metric, values in result_dict.items():
                writer.writerow(
                    {
                        "metric": metric,
                        "tile=16": values["tile=16"],
                        "tile=64": values["tile=64"],
                        "tile=dynamic": values["tile=dynamic"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        
    # save results to csv
    out_file = f"./dyn_tiling/figure_6_qwen_b64_raw.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "metric",
                "tile=16",
                "tile=64",
                "tile=dynamic",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for metric, values in result_dict_raw.items():
                writer.writerow(
                    {
                        "metric": metric,
                        "tile=16": values["tile=16"],
                        "tile=64": values["tile=64"],
                        "tile=dynamic": values["tile=dynamic"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
