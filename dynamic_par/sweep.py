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
    batch_list_b64 = [
        # batch with high stdev
        # {"start": 961, "end": 1024, "stdev": 1457},
        # {"start": 3239, "end": 3302, "stdev": 1374},
        # {"start": 1727, "end": 1790, "stdev": 1370},
        # batch with medium_high stdev
        # {"start": 3897, "end": 3960, "stdev": 1226},
        # {"start": 881, "end": 944, "stdev": 1226},
        # {"start": 1013, "end": 1076, "stdev": 1226},
        # # batch with similar stdev
        # {"start": 4007, "end": 4070, "stdev": 996},
        # {"start": 3123, "end": 3186, "stdev": 995},
        # {"start": 733, "end": 796, "stdev": 995},
        # # batch with medium_low stdev
        # {"start": 1505, "end": 1568, "stdev": 754},
        # {"start": 355, "end": 418, "stdev": 754},
        # {"start": 181, "end": 244, "stdev": 754},
        # # batch with lowest stdev
        {"start": 2019, "end": 2082, "stdev": 508},
        {"start": 4185, "end": 4248, "stdev": 484},
        {"start": 271, "end": 334, "stdev": 477},
    ]

    results = []
    for batch_spec in batch_list_b64:
        result_dict = {
            "stdev": batch_spec["stdev"],
            "start_idx": batch_spec["start"],
            "end_idx": batch_spec["end"],
        }
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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/static_coarse_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{static_coarse_cycles}\n")

        result_dict["static_coarse_cycles"] = static_coarse_cycles

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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/static_interleave_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{static_interleave_cycles}\n")

        result_dict["static_interleave_cycles"] = static_interleave_cycles

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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/dynamic_interleave_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{dynamic_cycles}\n")

        result_dict["dynamic_cycles"] = dynamic_cycles

        results.append(result_dict)

    # save results to csv
    out_file = f"./dynamic_par/batch64_sweep.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "batch",
                "unit_flops",
                "max_N",
                "tile_N",
                "stdev",
                "start_idx",
                "end_idx",
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
                        "batch": batch,
                        "unit_flops": compute_bw["qkt"],
                        "max_N": maxN,
                        "tile_N": tile_N,
                        "stdev": result["stdev"],
                        "start_idx": result["start_idx"],
                        "end_idx": result["end_idx"],
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
    batch_list_b16 = [
        # batch with super high stdev due to the small batch size
        {"start": 1487, "end": 1502, "stdev": 1974, "max_N": 8000},
        {"start": 3727, "end": 3742, "stdev": 1951, "max_N": 6528},
        {"start": 1501, "end": 1516, "stdev": 1884, "max_N": 8000},
        # batch with high stdev
        {"start": 2805, "end": 2820, "stdev": 1456, "max_N": 4672},
        {"start": 821, "end": 836, "stdev": 1455, "max_N": 4672},
        {"start": 985, "end": 1000, "stdev": 1454, "max_N": 4672},
        # batch with medium_high stdev
        # {"start": 4099, "end": 4114, "stdev": 1226, "max_N": 4672},
        # {"start": 1049, "end": 1064, "stdev": 1226, "max_N": 4672},
        # {"start": 75, "end": 90, "stdev": 1226, "max_N": 4672},
        # batch with similar stdev
        {"start": 3349, "end": 3364, "stdev": 987, "max_N": 4672},
        {"start": 3275, "end": 3290, "stdev": 987, "max_N": 4672},
        {"start": 3181, "end": 3196, "stdev": 986, "max_N": 4672},
        # batch with medium_low stdev
        # {"start": 1451, "end": 1466, "stdev": 754, "max_N": 4672},
        # {"start": 1851, "end": 1866, "stdev": 754, "max_N": 4672},
        # {"start": 1989, "end": 2004, "stdev": 752, "max_N": 4672},
        # batch with lowest stdev
        {"start": 3799, "end": 3814, "stdev": 480, "max_N": 4672},
        {"start": 2477, "end": 2492, "stdev": 479, "max_N": 4672},
        {"start": 1063, "end": 1078, "stdev": 479, "max_N": 4672},
        # batch with super low stdev due to the small batch size
        {"start": 1683, "end": 1698, "stdev": 198, "max_N": 2048},
        {"start": 1443, "end": 1458, "stdev": 198, "max_N": 2048},
        {"start": 1845, "end": 1860, "stdev": 174, "max_N": 2048},
    ]

    results = []
    for batch_spec in batch_list_b16:
        # ====== Cache config ======
        maxN = batch_spec["max_N"]
        tile_N = 32
        cache_row_offset_tiled = maxN // tile_N

        # ====== Data Creation ======
        k_cache = torch.zeros(batch, maxN, model_config.head_dim)
        v_cache = torch.zeros(batch, maxN, model_config.head_dim)

        result_dict = {
            "max_N": maxN,
            "stdev": batch_spec["stdev"],
            "start_idx": batch_spec["start"],
            "end_idx": batch_spec["end"],
        }
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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/static_coarse_b16_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{static_coarse_cycles}\n")

        result_dict["static_coarse_cycles"] = static_coarse_cycles

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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/static_interleave_b16_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{static_interleave_cycles}\n")

        result_dict["static_interleave_cycles"] = static_interleave_cycles

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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/dynamic_interleave_b16_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{dynamic_cycles}\n")

        result_dict["dynamic_cycles"] = dynamic_cycles

        results.append(result_dict)

    # save results to csv
    out_file = f"./dynamic_par/batch16_sweep.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "batch",
                "unit_flops",
                "max_N",
                "tile_N",
                "stdev",
                "start_idx",
                "end_idx",
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
                        "batch": batch,
                        "unit_flops": compute_bw["qkt"],
                        "max_N": result["max_N"],
                        "tile_N": tile_N,
                        "stdev": result["stdev"],
                        "start_idx": result["start_idx"],
                        "end_idx": result["end_idx"],
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
    batch_list_b80 = [
        # batch with high stdev
        # {"start": 981, "end": 1060, "stdev": 1413},
        {"start": 3227, "end": 3306, "stdev": 1334},
        {"start": 815, "end": 894, "stdev": 1310},
        # batch with medium_high stdev
        # {"start": 3537, "end": 3616, "stdev": 1226},
        # {"start": 4059, "end": 4138, "stdev": 1225},
        # {"start": 869, "end": 948, "stdev": 1224},
        # batch with similar stdev
        {"start": 4687, "end": 4766, "stdev": 987},
        {"start": 3989, "end": 4068, "stdev": 986},
        {"start": 1891, "end": 1970, "stdev": 987},
        # batch with medium_low stdev
        # {"start": 309, "end": 388, "stdev": 755},
        # {"start": 2115, "end": 2194, "stdev": 754},
        # {"start": 101, "end": 180, "stdev": 754},
        # batch with lowest stdev
        {"start": 4181, "end": 4260, "stdev": 562},
        {"start": 135, "end": 214, "stdev": 531},
        {"start": 2025, "end": 2104, "stdev": 529},
    ]

    results = []
    for batch_spec in batch_list_b80:
        result_dict = {
            "stdev": batch_spec["stdev"],
            "start_idx": batch_spec["start"],
            "end_idx": batch_spec["end"],
        }
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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/static_coarse_b80_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{static_coarse_cycles}\n")

        result_dict["static_coarse_cycles"] = static_coarse_cycles

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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/static_interleave_b80_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{static_interleave_cycles}\n")

        result_dict["static_interleave_cycles"] = static_interleave_cycles

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
            simulate_rust="full",  # "full", "timing", "serialize", None
            check_gold=False,
            save_graph=True,
        )
        with open(
            f"./dynamic_par/dynamic_interleave_b80_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"{dynamic_cycles}\n")

        result_dict["dynamic_cycles"] = dynamic_cycles

        results.append(result_dict)

    # save results to csv
    out_file = f"./dynamic_par/batch80_sweep.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "batch",
                "unit_flops",
                "max_N",
                "tile_N",
                "stdev",
                "start_idx",
                "end_idx",
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
                        "batch": batch,
                        "unit_flops": compute_bw["qkt"],
                        "max_N": maxN,
                        "tile_N": tile_N,
                        "stdev": result["stdev"],
                        "start_idx": result["start_idx"],
                        "end_idx": result["end_idx"],
                        "static_coarse_cycles": result["static_coarse_cycles"],
                        "static_interleave_cycles": result["static_interleave_cycles"],
                        "dynamic_cycles": result["dynamic_cycles"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def test_max_len_b64():
    batch_list_b64 = [
        # batch with high stdev
        {"start": 961, "end": 1024, "stdev": 1457},
        {"start": 3239, "end": 3302, "stdev": 1374},
        {"start": 1727, "end": 1790, "stdev": 1370},
        # batch with medium_high stdev
        {"start": 3897, "end": 3960, "stdev": 1226},
        {"start": 881, "end": 944, "stdev": 1226},
        {"start": 1013, "end": 1076, "stdev": 1226},
        # batch with similar stdev
        {"start": 4007, "end": 4070, "stdev": 996},
        {"start": 3123, "end": 3186, "stdev": 995},
        {"start": 733, "end": 796, "stdev": 995},
        # batch with medium_low stdev
        {"start": 1505, "end": 1568, "stdev": 754},
        {"start": 355, "end": 418, "stdev": 754},
        {"start": 181, "end": 244, "stdev": 754},
        # batch with lowest stdev
        {"start": 2019, "end": 2082, "stdev": 508},
        {"start": 4185, "end": 4248, "stdev": 484},
        {"start": 271, "end": 334, "stdev": 477},
    ]
    print("\n\n")
    for batch_spec in batch_list_b64:
        assert (
            64 == batch_spec["end"] - batch_spec["start"] + 1
        ), "Batch must be equal to high - low + 1"

        trace_file = f"./dynamic_par/azure_trace/b{64}/conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy"
        raw_np_arr = np.load(trace_file)
        np_arr_int = raw_np_arr.astype(np.int64)
        num_token_list = np_arr_int.tolist()
        print(
            f"conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy: {max(num_token_list)}"
        )

        """
        conv_stdev1457_0961_1024.npy: 4145
        conv_stdev1374_3239_3302.npy: 5020                                                                                                        
        conv_stdev1370_1727_1790.npy: 6445                                                                                                        
        conv_stdev1226_3897_3960.npy: 4098                                                                                                        
        conv_stdev1226_0881_0944.npy: 4126                                                                                                        
        conv_stdev1226_1013_1076.npy: 4122                                                                                                        
        conv_stdev0996_4007_4070.npy: 4088                                                                                                        
        conv_stdev0995_3123_3186.npy: 4974                                                                                                        
        conv_stdev0995_0733_0796.npy: 4096 
        conv_stdev0754_1505_1568.npy: 4083 
        conv_stdev0754_0355_0418.npy: 4083 
        conv_stdev0754_0181_0244.npy: 4082 
        conv_stdev0508_2019_2082.npy: 2739 
        conv_stdev0484_4185_4248.npy: 2740 
        conv_stdev0477_0271_0334.npy: 2522
        """


def test_max_len_b16():
    batch_list_b16 = [
        # batch with super high stdev due to the small batch size
        {"start": 1487, "end": 1502, "stdev": 1974},
        {"start": 3727, "end": 3742, "stdev": 1951},
        {"start": 1501, "end": 1516, "stdev": 1884},
        # batch with high stdev
        {"start": 2805, "end": 2820, "stdev": 1456},
        {"start": 821, "end": 836, "stdev": 1455},
        {"start": 985, "end": 1000, "stdev": 1454},
        # batch with medium_high stdev
        {"start": 4099, "end": 4114, "stdev": 1226},
        {"start": 1049, "end": 1064, "stdev": 1226},
        {"start": 75, "end": 90, "stdev": 1226},
        # batch with similar stdev
        {"start": 3349, "end": 3364, "stdev": 987},
        {"start": 3275, "end": 3290, "stdev": 987},
        {"start": 3181, "end": 3196, "stdev": 986},
        # batch with medium_low stdev
        {"start": 1451, "end": 1466, "stdev": 754},
        {"start": 1851, "end": 1866, "stdev": 754},
        {"start": 1989, "end": 2004, "stdev": 752},
        # batch with lowest stdev
        {"start": 3799, "end": 3814, "stdev": 480},
        {"start": 2477, "end": 2492, "stdev": 479},
        {"start": 1063, "end": 1078, "stdev": 479},
        # batch with super low stdev due to the small batch size
        {"start": 1683, "end": 1698, "stdev": 198},
        {"start": 1443, "end": 1458, "stdev": 198},
        {"start": 1845, "end": 1860, "stdev": 174},
    ]

    print("\n\n")
    for batch_spec in batch_list_b16:
        assert (
            16 == batch_spec["end"] - batch_spec["start"] + 1
        ), "Batch must be equal to high - low + 1"

        trace_file = f"./dynamic_par/azure_trace/b{16}/conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy"
        raw_np_arr = np.load(trace_file)
        np_arr_int = raw_np_arr.astype(np.int64)
        num_token_list = np_arr_int.tolist()
        print(
            f"conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy: {max(num_token_list)}"
        )
        """ 
        conv_stdev1974_1487_1502.npy: 7930
        conv_stdev1951_3727_3742.npy: 6472
        conv_stdev1884_1501_1516.npy: 7930
        conv_stdev1456_2805_2820.npy: 4623
        conv_stdev1455_0821_0836.npy: 4119
        conv_stdev1454_0985_1000.npy: 4122
        conv_stdev1226_4099_4114.npy: 4083
        conv_stdev1226_1049_1064.npy: 4122
        conv_stdev1226_0075_0090.npy: 4094
        conv_stdev0987_3349_3364.npy: 4079
        conv_stdev0987_3275_3290.npy: 4080
        conv_stdev0986_3181_3196.npy: 4076
        conv_stdev0754_1451_1466.npy: 4073
        conv_stdev0754_1851_1866.npy: 4072
        conv_stdev0752_1989_2004.npy: 2739
        conv_stdev0480_3799_3814.npy: 2255
        conv_stdev0479_2477_2492.npy: 2202
        conv_stdev0479_1063_1078.npy: 1320
        conv_stdev0198_1683_1698.npy: 1155
        conv_stdev0198_1443_1458.npy: 1785
        conv_stdev0174_1845_1860.npy: 1145
        """


def test_max_len_b80():
    batch_list_b80 = [
        # batch with high stdev
        {"start": 981, "end": 1060, "stdev": 1413},
        {"start": 3227, "end": 3306, "stdev": 1334},
        {"start": 815, "end": 894, "stdev": 1310},
        # batch with medium_high stdev
        {"start": 3537, "end": 3616, "stdev": 1226},
        {"start": 4059, "end": 4138, "stdev": 1225},
        {"start": 869, "end": 948, "stdev": 1224},
        # batch with similar stdev
        {"start": 4687, "end": 4766, "stdev": 987},
        {"start": 3989, "end": 4068, "stdev": 986},
        {"start": 1891, "end": 1970, "stdev": 987},
        # batch with medium_low stdev
        {"start": 309, "end": 388, "stdev": 755},
        {"start": 2115, "end": 2194, "stdev": 754},
        {"start": 101, "end": 180, "stdev": 754},
        # batch with lowest stdev
        {"start": 4181, "end": 4260, "stdev": 562},
        {"start": 135, "end": 214, "stdev": 531},
        {"start": 2025, "end": 2104, "stdev": 529},
    ]

    print("\n\n")
    for batch_spec in batch_list_b80:
        assert (
            80 == batch_spec["end"] - batch_spec["start"] + 1
        ), "Batch must be equal to high - low + 1"

        trace_file = f"./dynamic_par/azure_trace/b{80}/conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy"
        raw_np_arr = np.load(trace_file)
        np_arr_int = raw_np_arr.astype(np.int64)
        num_token_list = np_arr_int.tolist()
        print(
            f"conv_stdev{batch_spec['stdev']:04d}_{batch_spec['start']:04d}_{batch_spec['end']:04d}.npy: {max(num_token_list)}"
        )

    """
    conv_stdev1413_0981_1060.npy: 4122
    conv_stdev1334_3227_3306.npy: 5069
    conv_stdev1310_0815_0894.npy: 4132
    conv_stdev1226_3537_3616.npy: 6472
    conv_stdev1225_4059_4138.npy: 4088
    conv_stdev1224_0869_0948.npy: 4126
    conv_stdev0987_4687_4766.npy: 4782
    conv_stdev0986_3989_4068.npy: 4090
    conv_stdev0987_1891_1970.npy: 4091
    conv_stdev0755_0309_0388.npy: 4088
    conv_stdev0754_2115_2194.npy: 4125
    conv_stdev0754_0101_0180.npy: 4107
    conv_stdev0562_4181_4260.npy: 2888
    conv_stdev0531_0135_0214.npy: 4082
    conv_stdev0529_2025_2104.npy: 2739
    """
