from typing import Dict
from networkx import MultiDiGraph

from dynamic_par.flashattn import build_flashattn_graph
from rewrite.broadcast import infer_broadcast
from sim import HBMConfig, SimConfig, serialize, simulate
from step_py.ops import *
from step_py.utility_ops import *
from step_py.functions import accum_fn, init_fn, map_fn
from dataclasses import dataclass
import random
from utils.draw_graph import save_graph_format
from utils.gold_checking import reconstruct_numpy
import numpy as np


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


def build_static_par(
    step_graph: MultiDiGraph,
    model_config,
    query: torch.Tensor,  # [B*query_per_kvhead, D] (query_per_kvhead = 4 for Qwen, 8 for Mixtral)
    key: torch.Tensor,  # [B, D]
    value: torch.Tensor,  # [B, D]
    k_cache: torch.Tensor,  # [B * N, D]
    v_cache: torch.Tensor,  # [B * N, D]
    idx: torch.Tensor,  # [B]
    seq_len: torch.Tensor,  # [B]
    offset: torch.Tensor,  # [B]
    par_factor: int,
    metadata_fifo_depth: int,
    cache_write_back_fifo_depth: int,
    cache_row_offset_tiled: int,  # N
    tile_N: int,
    compute_bw: Dict[str, int],
    mock_bf16: bool = False,
    par_dispatch: int = 1,
    channel_dict: Dict[int, int] = {},
) -> List[RandomOffChipStore]:
    B = idx.shape[0]
    query_per_kvhead = model_config.query_per_kvhead
    head_dim = model_config.head_dim

    # ------------ Stage 1: Load the query, key, value, metadata ------------
    # output shape: [1, B, 1] (tile: [query_per_kvhead, D])
    load_q = OffChipLoad(
        underlying=query,
        stride=(1, 1),
        out_shape_tiled=(B, 1),
        tile_row=query_per_kvhead,
        tile_col=head_dim,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    # output shape: [1, B, 1] (tile: [1,D])
    load_k = OffChipLoad(
        underlying=query,
        stride=(1, 1),
        out_shape_tiled=(B, 1),
        tile_row=1,
        tile_col=head_dim,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    # output shape: [1, B, 1] (tile: [1,D])
    load_v = OffChipLoad(
        underlying=query,
        stride=(1, 1),
        out_shape_tiled=(B, 1),
        tile_row=1,
        tile_col=head_dim,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    # output shape: [1, B] (tile: [1,1])
    load_idx = MetadataGen(
        tensor=idx,
    )

    # output shape: [1, B] (tile: [1,1])
    load_seq_len = MetadataGen(
        tensor=seq_len,
    )

    # output shape: [1, B] (tile: [1,1])
    load_offset = MetadataGen(
        tensor=offset,
    )

    # ------------ Stage 2: Generate control data for parallelization ------------
    # Don't need it for static parallelization

    # ------------ Stage 3: Partition data to parallelize ------------
    # input stream shape:   [1, B, 1] (tile: [query_per_kvhead, D])
    #       (after flatten) [1, B]    (tile: [query_per_kvhead, D])
    # output stream shape:  [B//par_factor,1] x par_factor (tile: [query_per_kvhead, D])
    partition_query = Parallelize(
        graph=step_graph,
        input=Flatten(graph=step_graph, input=load_q, min_rank=1, max_rank=2),
        parallelize_rank=1,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    # input stream shape:   [1, B, 1] (tile: [1, D])
    #       (after flatten) [1, B]    (tile: [1, D])
    # output stream shape:  [B//par_factor,1] x par_factor (tile: [1, D])
    partition_key = Parallelize(
        graph=step_graph,
        input=Flatten(graph=step_graph, input=load_k, min_rank=1, max_rank=2),
        parallelize_rank=1,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    # input stream shape:   [1, B, 1] (tile: [1, D])
    #       (after flatten) [1, B]    (tile: [1, D])
    # output stream shape:  [B//par_factor,1] x par_factor (tile: [1, D])
    partition_value = Parallelize(
        graph=step_graph,
        input=Flatten(graph=step_graph, input=load_v, min_rank=1, max_rank=2),
        parallelize_rank=1,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    # input stream shape:   [1, B] (tile: [1, 1])
    #       (after flatten) [B]    (tile: [1, 1])
    # output stream shape:  [B//par_factor] x par_factor (tile: [1, 1])
    partition_idx = Parallelize(
        graph=step_graph,
        input=Flatten(graph=step_graph, input=load_idx, min_rank=0, max_rank=1),
        parallelize_rank=0,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    # input stream shape:   [1, B] (tile: [1, 1])
    #       (after flatten) [B]    (tile: [1, 1])
    # output stream shape:  [B//par_factor] x par_factor (tile: [1, 1])
    partition_seq_len = Parallelize(
        graph=step_graph,
        input=Flatten(graph=step_graph, input=load_seq_len, min_rank=0, max_rank=1),
        parallelize_rank=0,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    # input stream shape:   [1, B] (tile: [1, 1])
    #       (after flatten) [B]    (tile: [1, 1])
    # output stream shape:  [B//par_factor] x par_factor (tile: [1, 1])
    partition_offset = Parallelize(
        graph=step_graph,
        input=Flatten(graph=step_graph, input=load_offset, min_rank=0, max_rank=1),
        parallelize_rank=0,
        switch_cycles=[1] * par_factor,
        write_back_mu=False,
        num_consumers=par_factor,
    )

    broadcast_idx_list = [
        Broadcast(graph=step_graph, input=(partition_idx, i), num_consumers=5)
        for i in range(par_factor)
    ]
    broadcast_seq_len_list = [
        Broadcast(graph=step_graph, input=(partition_seq_len, i), num_consumers=6)
        for i in range(par_factor)
    ]
    broadcast_offset_list = [
        Broadcast(graph=step_graph, input=(partition_offset, i), num_consumers=2)
        for i in range(par_factor)
    ]
    for i in range(par_factor):
        channel_dict[broadcast_idx_list[i].instance_id] = metadata_fifo_depth
        channel_dict[broadcast_seq_len_list[i].instance_id] = metadata_fifo_depth
        channel_dict[broadcast_offset_list[i].instance_id] = metadata_fifo_depth

    output_list = []
    for i in range(par_factor):
        _, new_channel_dict, store_output = build_flashattn_graph(
            par_region_idx=i,
            step_graph=step_graph,
            model_config=model_config,
            query=(partition_query, i),
            key=(partition_key, i),
            value=(partition_value, i),
            k_cache_underlying=k_cache,
            v_cache_underlying=v_cache,
            output_underlying=torch.zeros(
                (B * model_config.query_per_kvhead, model_config.head_dim),
                dtype=torch.float32,
            ),  # [B * query_per_kvhead, D]
            idx_metadata=broadcast_idx_list[i],
            seq_len_metadata=broadcast_seq_len_list[i],
            offset_metadata=broadcast_offset_list[i],
            cache_row_offset_tiled=cache_row_offset_tiled,
            cache_write_back_fifo_depth=cache_write_back_fifo_depth,
            tile_N=tile_N,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
            compute_bw=compute_bw,
            channel_dict=channel_dict,
        )
        output_list.append(store_output)
        channel_dict = new_channel_dict

    return output_list


def run_static_par(
    batch: int,
    cache_row_offset_tiled: int,
    tile_N: int,
    metadata_fifo_depth: int,
    cache_write_back_fifo_depth: int,
    model_config,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    offset: torch.Tensor,
    compute_bw: Dict[str, int],
    mock_bf16: bool,
    simulate_rust: Optional[str],
    check_gold: bool,
    save_graph: bool,
    logging: Optional[str] = None,
):
    channel_dict = {}

    # ----------- Create input ------------------
    query = torch.randn(batch * model_config.query_per_kvhead, model_config.head_dim)
    key = torch.randn(batch, model_config.head_dim)
    value = torch.randn(batch, model_config.head_dim)

    # Build graph
    step_graph = MultiDiGraph()
    output_list = build_static_par(
        step_graph=step_graph,
        model_config=model_config,
        query=query,
        key=key,
        value=value,
        k_cache=k_cache,
        v_cache=v_cache,
        idx=torch.arange(batch),
        seq_len=seq_len,
        offset=offset,
        par_factor=4,
        metadata_fifo_depth=metadata_fifo_depth,
        cache_write_back_fifo_depth=cache_write_back_fifo_depth,
        cache_row_offset_tiled=cache_row_offset_tiled,
        tile_N=tile_N,
        compute_bw=compute_bw,
        mock_bf16=mock_bf16,
        par_dispatch=1,
        channel_dict=channel_dict,
    )

    step_graph = infer_broadcast(step_graph)

    if save_graph:
        OUTPUT_FILENAME = "static_par"
        save_graph_format(step_graph, OUTPUT_FILENAME, ["svg"])

    # Run simulation
    cycles = 0
    duration_ms = 0
    duration_s = 0

    if simulate_rust in ["full", "timing"]:
        hbm_config = HBMConfig(64, 32, 2, 2, 1, 14)
        sim_config = SimConfig(
            channel_depth=1,
            functional_sim=simulate_rust == "full",
            mock_bf16=mock_bf16,
            config_dict=channel_dict,
        )
        if logging is None:
            cycles, duration_ms, duration_s = simulate(
                step_graph,
                False,  # logging
                hbm_config,
                sim_config,
                "./graph.pb",
            )
        else:
            assert isinstance(logging, str), "Logging must be a string path"
            cycles, duration_ms, duration_s = simulate(
                step_graph,
                True,  # logging
                hbm_config,
                sim_config,
                "./graph.pb",
                logging,
            )
    elif simulate_rust == "serialize":
        serialize(step_graph, "./graph.pb", False)

    # Gold check
    if check_gold:
        # Q: [B * query_per_kvhead, head_dim] -> [B, query_per_kvhead, head_dim]

        # Create K cache:
        # - [B, head_dim] -> [B, 1, head_dim]
        # - [B * maxN, head_dim]

        simulated_output = torch.zeros(
            (batch * model_config.query_per_kvhead, model_config.head_dim),
            dtype=torch.float32,
        )
        for output in output_list:
            simulated_output += reconstruct_numpy(output.store_file_name)

        print(simulated_output.shape)

    return cycles


def test_static_par():
    # ====== Model config ======
    model_config = Qwen30B()

    # ====== Input config ======
    batch = 16

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
    maxN = 4160
    tile_N = 32
    cache_row_offset_tiled = maxN // tile_N

    # ====== Data Creation ======
    k_cache = torch.zeros(batch, maxN, model_config.head_dim)
    v_cache = torch.zeros(batch, maxN, model_config.head_dim)

    random.seed(42)
    # num_token_list = [random.randint(8, 30) for _ in range(batch)]
    # var = "h"
    # low = 77
    # high = 92

    # var = "h"
    # low = 13
    # high = 76

    var = "m"
    low = 179
    high = 194

    # var = "m"
    # low = 115
    # high = 178
    trace_file = f"dynamic_par/azure_trace/conv_{var}_{low:03d}_{high:03d}.npy"
    raw_np_arr = np.load(trace_file)
    np_arr_int = raw_np_arr.astype(np.int64)
    num_token_list = np_arr_int.tolist()
    print(f"num_token_list: {num_token_list}")

    seq_len_tiled = torch.tensor(
        [(x + 1 + tile_N - 1) // tile_N for x in num_token_list]
    )

    offset = torch.tensor([x % tile_N for x in num_token_list])

    # ====== Initialize KV cache ======
    for i in range(batch):
        k_cache[i, : num_token_list[i]] = torch.randn(
            num_token_list[i], model_config.head_dim
        )
        v_cache[i, : num_token_list[i]] = torch.randn(
            num_token_list[i], model_config.head_dim
        )

    run_static_par(
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
