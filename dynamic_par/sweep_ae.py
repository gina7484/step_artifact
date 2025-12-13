import csv
from dynamic_par.static_coarse_parallel import (
    run_static_coarse_par,
    run_static_coarse_par_b80,
)
from dynamic_par.static_parallel import run_static_par
from dynamic_par.dynamic_parallel import run_dynmic_par
from dataclasses import dataclass
import torch
import numpy as np
import random
import math


@dataclass
class Mixtral8x7B:
    hidden_dim = 4096
    head_dim = 128
    num_heads = 32
    num_kv_heads = 8
    query_per_kvhead = 4  # Should be (num_heads / num_kv_heads)


@dataclass
class Qwen30B:
    hidden_dim = 2048
    head_dim = 64
    num_heads = 32
    num_kv_heads = 4
    query_per_kvhead = 8  # Should be (num_heads / num_kv_heads)


def test_b64_sweep():
    # ====== Model config ======
    model_config = Qwen30B()

    # ====== Channel config ======
    metadata_fifo_depth = 16
    cache_write_back_fifo_depth = 4

    # ====== Compute bandwidth config ======
    compute_bw = {
        "qkt": 2048,
        "exp": 2048,
        "multv": 2048,
        "tile_wise_rowsum": 2048,
        "intra_tile_rowsum": 2048,
        "softmax_div": 2048,
    }

    # ====== Cache config ======
    maxN = 6464
    tile_N = 32
    cache_row_offset_tiled = maxN // tile_N

    # ====== Input config ======
    batch = 64

    # ====== Data Creation ======
    k_cache = torch.zeros(batch, maxN, model_config.head_dim)
    v_cache = torch.zeros(batch, maxN, model_config.head_dim)

    random.seed(42)
    # num_token_list = [random.randint(8, 30) for _ in range(batch)]

    # ---- config for batch 64 ----
    batch_list_b64 = {
        "high": [
            # batch with high stdev
            {"start": 961, "end": 1024, "stdev": 1457},
            {"start": 3239, "end": 3302, "stdev": 1374},
            {"start": 1727, "end": 1790, "stdev": 1370},
        ],
        "med": [
            # batch with similar stdev
            {"start": 4007, "end": 4070, "stdev": 996},
            {"start": 3123, "end": 3186, "stdev": 995},
            {"start": 733, "end": 796, "stdev": 995},
        ],
        "low": [
            # batch with lowest stdev
            {"start": 2019, "end": 2082, "stdev": 508},
            {"start": 4185, "end": 4248, "stdev": 484},
            {"start": 271, "end": 334, "stdev": 477},
        ],
    }

    check_intermediate = False

    results = []

    raw_results = []

    for kv_length_var, batch_list in batch_list_b64.items():

        result_dict = {}
        result_dict["kv_length_var"] = kv_length_var

        static_coarse_cycles_list = []
        static_interleave_cycles_list = []
        dynamic_cycles_list = []

        for batch_spec in batch_list:
            raw_result_dict = {}
            raw_result_dict["kv_length_var"] = kv_length_var

            assert (
                batch == batch_spec["end"] - batch_spec["start"] + 1
            ), "Batch must be equal to high - low + 1"

            trace_file = f"./dynamic_par/azure_trace/b{batch}/conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy"
            raw_np_arr = np.load(trace_file)
            np_arr_int = raw_np_arr.astype(np.int64)
            num_token_list = np_arr_int.tolist()
            print(f"num_token_list: {num_token_list}")

            seq_len_tiled = torch.tensor(
                [(x + 1 + tile_N - 1) // tile_N for x in num_token_list]
            )

            offset = torch.tensor([x % tile_N for x in num_token_list])

            # ====== Initialize KV cache ======
            # k_cache: [batch, maxN, head_dim]
            # v_cache: [batch, maxN, head_dim]
            for i in range(batch):
                k_cache[i, : num_token_list[i]] = torch.randn(
                    num_token_list[i], model_config.head_dim
                )
                v_cache[i, : num_token_list[i]] = torch.randn(
                    num_token_list[i], model_config.head_dim
                )

            static_coarse_cycles = run_static_coarse_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                output_underlying=torch.zeros(
                    (batch * model_config.query_per_kvhead, model_config.head_dim),
                    dtype=torch.float32,
                ),  # [B * query_per_kvhead, D]
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )
            if check_intermediate:
                with open(
                    f"./dynamic_par/static_coarse_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{static_coarse_cycles}\n")

            static_interleave_cycles = run_static_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/static_interleave_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{static_interleave_cycles}\n")

            dynamic_cycles = run_dynmic_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )
            if check_intermediate:
                with open(
                    f"./dynamic_par/dynamic_interleave_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{dynamic_cycles}\n")

            static_coarse_cycles_list.append(static_coarse_cycles / dynamic_cycles)

            static_interleave_cycles_list.append(
                static_interleave_cycles / dynamic_cycles
            )

            dynamic_cycles_list.append(dynamic_cycles / dynamic_cycles)

            raw_result_dict["stdev"] = batch_spec["stdev"]
            raw_result_dict["static_coarse_cycles"] = static_coarse_cycles
            raw_result_dict["static_interleave_cycles"] = static_interleave_cycles
            raw_result_dict["dynamic_cycles"] = dynamic_cycles
            raw_results.append(raw_result_dict)

        normalized_static_coarse = math.prod(static_coarse_cycles_list) ** (
            1 / len(static_coarse_cycles_list)
        )
        normalized_static_interleave = math.prod(static_interleave_cycles_list) ** (
            1 / len(static_interleave_cycles_list)
        )
        normalized_dynamic = math.prod(dynamic_cycles_list) ** (
            1 / len(dynamic_cycles_list)
        )

        result_dict["static_coarse_cycles"] = normalized_static_coarse
        result_dict["static_interleave_cycles"] = normalized_static_interleave
        result_dict["dynamic_cycles"] = normalized_dynamic
        results.append(result_dict)

    # save results to csv
    out_file = f"./dynamic_par/batch64_sweep_ae.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "kv_length_var",
                "static_coarse_cycles",
                "static_interleave_cycles",
                "dynamic_cycles",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in results:
                writer.writerow(
                    {
                        "kv_length_var": result["kv_length_var"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

    # save results to csv
    out_file = f"./dynamic_par/batch64_sweep_ae_raw.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "kv_length_var",
                "stdev",
                "static_coarse_cycles",
                "static_interleave_cycles",
                "dynamic_cycles",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in raw_results:
                writer.writerow(
                    {
                        "kv_length_var": result["kv_length_var"],
                        "stdev": result["stdev"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def test_b16_sweep():
    # ====== Model config ======
    model_config = Qwen30B()

    # ====== Channel config ======
    metadata_fifo_depth = 16
    cache_write_back_fifo_depth = 4

    # ====== Compute bandwidth config ======
    compute_bw = {
        "qkt": 2048,
        "exp": 2048,
        "multv": 2048,
        "tile_wise_rowsum": 2048,
        "intra_tile_rowsum": 2048,
        "softmax_div": 2048,
    }
    # ====== Input config ======
    batch = 16

    random.seed(42)
    # num_token_list = [random.randint(8, 30) for _ in range(batch)]

    # ---- config for batch 64 ----
    batch_list_b16 = {
        "high": [
            # batch with high stdev
            {"start": 2805, "end": 2820, "stdev": 1456, "max_N": 4672},
            {"start": 821, "end": 836, "stdev": 1455, "max_N": 4672},
            {"start": 985, "end": 1000, "stdev": 1454, "max_N": 4672},
        ],
        "med": [
            # batch with similar stdev
            {"start": 3349, "end": 3364, "stdev": 987, "max_N": 4672},
            {"start": 3275, "end": 3290, "stdev": 987, "max_N": 4672},
            {"start": 3181, "end": 3196, "stdev": 986, "max_N": 4672},
        ],
        "low": [
            # batch with low stdev
            {"start": 3799, "end": 3814, "stdev": 480, "max_N": 4672},
            {"start": 2477, "end": 2492, "stdev": 479, "max_N": 4672},
            {"start": 1063, "end": 1078, "stdev": 479, "max_N": 4672},
        ],
    }

    check_intermediate = False

    results = []

    raw_results = []

    for kv_length_var, batch_list in batch_list_b16.items():
        result_dict = {}
        result_dict["kv_length_var"] = kv_length_var

        static_coarse_cycles_list = []
        static_interleave_cycles_list = []
        dynamic_cycles_list = []

        for batch_spec in batch_list:
            raw_result_dict = {}
            raw_result_dict["kv_length_var"] = kv_length_var

            # ====== Cache config ======
            maxN = batch_spec["max_N"]
            tile_N = 32
            cache_row_offset_tiled = maxN // tile_N

            # ====== Data Creation ======
            k_cache = torch.zeros(batch, maxN, model_config.head_dim)
            v_cache = torch.zeros(batch, maxN, model_config.head_dim)

            assert (
                batch == batch_spec["end"] - batch_spec["start"] + 1
            ), "Batch must be equal to high - low + 1"

            trace_file = f"./dynamic_par/azure_trace/b{batch}/conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy"
            raw_np_arr = np.load(trace_file)
            np_arr_int = raw_np_arr.astype(np.int64)
            num_token_list = np_arr_int.tolist()
            print(f"num_token_list: {num_token_list}")

            seq_len_tiled = torch.tensor(
                [(x + 1 + tile_N - 1) // tile_N for x in num_token_list]
            )

            offset = torch.tensor([x % tile_N for x in num_token_list])

            # ====== Initialize KV cache ======
            # k_cache: [batch, maxN, head_dim]
            # v_cache: [batch, maxN, head_dim]
            for i in range(batch):
                k_cache[i, : num_token_list[i]] = torch.randn(
                    num_token_list[i], model_config.head_dim
                )
                v_cache[i, : num_token_list[i]] = torch.randn(
                    num_token_list[i], model_config.head_dim
                )

            static_coarse_cycles = run_static_coarse_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                output_underlying=torch.zeros(
                    (batch * model_config.query_per_kvhead, model_config.head_dim),
                    dtype=torch.float32,
                ),  # [B * query_per_kvhead, D]
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/static_coarse_b16_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{static_coarse_cycles}\n")

            static_interleave_cycles = run_static_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/static_interleave_b16_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{static_interleave_cycles}\n")

            dynamic_cycles = run_dynmic_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/dynamic_interleave_b16_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{dynamic_cycles}\n")

            static_coarse_cycles_list.append(static_coarse_cycles / dynamic_cycles)

            static_interleave_cycles_list.append(
                static_interleave_cycles / dynamic_cycles
            )

            dynamic_cycles_list.append(dynamic_cycles / dynamic_cycles)

            raw_result_dict["stdev"] = batch_spec["stdev"]
            raw_result_dict["static_coarse_cycles"] = static_coarse_cycles
            raw_result_dict["static_interleave_cycles"] = static_interleave_cycles
            raw_result_dict["dynamic_cycles"] = dynamic_cycles
            raw_results.append(raw_result_dict)

        normalized_static_coarse = math.prod(static_coarse_cycles_list) ** (
            1 / len(static_coarse_cycles_list)
        )
        normalized_static_interleave = math.prod(static_interleave_cycles_list) ** (
            1 / len(static_interleave_cycles_list)
        )
        normalized_dynamic = math.prod(dynamic_cycles_list) ** (
            1 / len(dynamic_cycles_list)
        )

        result_dict["static_coarse_cycles"] = normalized_static_coarse
        result_dict["static_interleave_cycles"] = normalized_static_interleave
        result_dict["dynamic_cycles"] = normalized_dynamic

        results.append(result_dict)

    # save results to csv
    out_file = f"./dynamic_par/batch16_sweep_ae.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "kv_length_var",
                "static_coarse_cycles",
                "static_interleave_cycles",
                "dynamic_cycles",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in results:
                writer.writerow(
                    {
                        "kv_length_var": result["kv_length_var"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

    # save results to csv
    out_file = f"./dynamic_par/batch16_sweep_ae_raw.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "kv_length_var",
                "stdev",
                "static_coarse_cycles",
                "static_interleave_cycles",
                "dynamic_cycles",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in raw_results:
                writer.writerow(
                    {
                        "kv_length_var": result["kv_length_var"],
                        "stdev": result["stdev"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def test_b64_b16_sweep():
    # ====== Model config ======
    model_config = Qwen30B()

    # ====== Channel config ======
    metadata_fifo_depth = 16
    cache_write_back_fifo_depth = 4

    # ====== Compute bandwidth config ======
    compute_bw = {
        "qkt": 2048,
        "exp": 2048,
        "multv": 2048,
        "tile_wise_rowsum": 2048,
        "intra_tile_rowsum": 2048,
        "softmax_div": 2048,
    }

    # ====== Cache config ======
    maxN = 6528
    tile_N = 32
    cache_row_offset_tiled = maxN // tile_N

    # ====== Input config ======
    batch = 80

    # ====== Data Creation ======
    k_cache = torch.zeros(batch, maxN, model_config.head_dim)
    v_cache = torch.zeros(batch, maxN, model_config.head_dim)

    random.seed(42)
    # num_token_list = [random.randint(8, 30) for _ in range(batch)]

    # ---- config for batch 80 ----
    batch_list_b80 = {
        "high": [
            # batch with high stdev
            {"start": 981, "end": 1060, "stdev": 1413},
            {"start": 3227, "end": 3306, "stdev": 1334},
            {"start": 815, "end": 894, "stdev": 1310},
        ],
        "med": [
            # batch with similar stdev
            {"start": 4687, "end": 4766, "stdev": 987},
            {"start": 3989, "end": 4068, "stdev": 986},
            {"start": 1891, "end": 1970, "stdev": 987},
        ],
        "low": [
            # batch with lowest stdev
            {"start": 4181, "end": 4260, "stdev": 562},
            {"start": 135, "end": 214, "stdev": 531},
            {"start": 2025, "end": 2104, "stdev": 529},
        ],
    }

    check_intermediate = False

    results = []

    raw_results = []

    for kv_length_var, batch_list in batch_list_b80.items():
        result_dict = {}
        result_dict["kv_length_var"] = kv_length_var

        static_coarse_cycles_list = []
        static_interleave_cycles_list = []
        dynamic_cycles_list = []

        for batch_spec in batch_list:
            raw_result_dict = {}
            raw_result_dict["kv_length_var"] = kv_length_var

            assert (
                batch == batch_spec["end"] - batch_spec["start"] + 1
            ), "Batch must be equal to high - low + 1"

            trace_file = f"./dynamic_par/azure_trace/b{batch}/conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy"
            raw_np_arr = np.load(trace_file)
            np_arr_int = raw_np_arr.astype(np.int64)
            num_token_list = np_arr_int.tolist()
            print(f"num_token_list: {num_token_list}")

            seq_len_tiled = torch.tensor(
                [(x + 1 + tile_N - 1) // tile_N for x in num_token_list]
            )

            offset = torch.tensor([x % tile_N for x in num_token_list])

            # ====== Initialize KV cache ======
            # k_cache: [batch, maxN, head_dim]
            # v_cache: [batch, maxN, head_dim]
            for i in range(batch):
                k_cache[i, : num_token_list[i]] = torch.randn(
                    num_token_list[i], model_config.head_dim
                )
                v_cache[i, : num_token_list[i]] = torch.randn(
                    num_token_list[i], model_config.head_dim
                )

            static_coarse_cycles = run_static_coarse_par_b80(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                output_underlying=torch.zeros(
                    (batch * model_config.query_per_kvhead, model_config.head_dim),
                    dtype=torch.float32,
                ),  # [B * query_per_kvhead, D]
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/static_coarse_b80_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{static_coarse_cycles}\n")

            static_interleave_cycles = run_static_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/static_interleave_b80_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{static_interleave_cycles}\n")

            dynamic_cycles = run_dynmic_par(
                batch=batch,
                cache_row_offset_tiled=cache_row_offset_tiled,
                tile_N=tile_N,
                metadata_fifo_depth=metadata_fifo_depth,
                cache_write_back_fifo_depth=cache_write_back_fifo_depth,
                model_config=model_config,
                k_cache=k_cache.reshape(batch * maxN, model_config.head_dim),
                v_cache=v_cache.reshape(batch * maxN, model_config.head_dim),
                seq_len=seq_len_tiled,
                offset=offset,
                compute_bw=compute_bw,
                mock_bf16=True,
                simulate_rust="timing",  # "full", "timing", "serialize", None
                check_gold=False,
                save_graph=True,
            )

            if check_intermediate:
                with open(
                    f"./dynamic_par/dynamic_interleave_b80_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{dynamic_cycles}\n")

            static_coarse_cycles_list.append(static_coarse_cycles / dynamic_cycles)

            static_interleave_cycles_list.append(
                static_interleave_cycles / dynamic_cycles
            )

            dynamic_cycles_list.append(dynamic_cycles / dynamic_cycles)

            raw_result_dict["stdev"] = batch_spec["stdev"]
            raw_result_dict["static_coarse_cycles"] = static_coarse_cycles
            raw_result_dict["static_interleave_cycles"] = static_interleave_cycles
            raw_result_dict["dynamic_cycles"] = dynamic_cycles
            raw_results.append(raw_result_dict)

        normalized_static_coarse = math.prod(static_coarse_cycles_list) ** (
            1 / len(static_coarse_cycles_list)
        )
        normalized_static_interleave = math.prod(static_interleave_cycles_list) ** (
            1 / len(static_interleave_cycles_list)
        )
        normalized_dynamic = math.prod(dynamic_cycles_list) ** (
            1 / len(dynamic_cycles_list)
        )

        result_dict["static_coarse_cycles"] = normalized_static_coarse
        result_dict["static_interleave_cycles"] = normalized_static_interleave
        result_dict["dynamic_cycles"] = normalized_dynamic

        results.append(result_dict)

    # save results to csv
    out_file = f"./dynamic_par/batch80_sweep_ae.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "kv_length_var",
                "static_coarse_cycles",
                "static_interleave_cycles",
                "dynamic_cycles",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in results:
                writer.writerow(
                    {
                        "kv_length_var": result["kv_length_var"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

    # save results to csv
    out_file = f"./dynamic_par/batch80_sweep_ae_raw.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "kv_length_var",
                "stdev",
                "static_coarse_cycles",
                "static_interleave_cycles",
                "dynamic_cycles",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in raw_results:
                writer.writerow(
                    {
                        "kv_length_var": result["kv_length_var"],
                        "stdev": result["stdev"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


