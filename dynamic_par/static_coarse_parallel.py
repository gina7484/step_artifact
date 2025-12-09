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


def build_static_coarse_par(
    step_graph: MultiDiGraph,
    model_config,
    query: List[
        torch.Tensor
    ],  # [16*query_per_kvhead, D] x par_factor(query_per_kvhead = 4 for Qwen, 8 for Mixtral)
    key: List[torch.Tensor],  # [16, D] x par_factor
    value: List[torch.Tensor],  # [16, D] x par_factor
    k_cache: torch.Tensor,  # [B * N, D]
    v_cache: torch.Tensor,  # [B * N, D]
    output_underlying: torch.Tensor,  # [B * query_per_kvhead, D]
    idx: torch.Tensor,  # [16] x par_factor
    seq_len: torch.Tensor,  # [16] x par_factor
    offset: torch.Tensor,  # [16] x par_factor
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
    query_per_kvhead = model_config.query_per_kvhead
    head_dim = model_config.head_dim

    # ------------ Stage 1: Load the query, key, value, metadata ------------
    # output shape: [1, 16, 1] x par_factor (tile: [query_per_kvhead, D])
    # (after flatten): [16, 1] x par_factor
    load_q = [
        Flatten(
            graph=step_graph,
            input=OffChipLoad(
                underlying=query[i],
                stride=(1, 1),
                out_shape_tiled=(idx[i].shape[0], 1),
                tile_row=query_per_kvhead,
                tile_col=head_dim,
                par_dispatch=par_dispatch,
                mock_bf16=mock_bf16,
            ),
            min_rank=1,
            max_rank=2,
        )
        for i in range(par_factor)
    ]

    # output shape: [1, 16, 1] x par_factor (tile: [1,D])
    # (after flatten): [16, 1] x par_factor
    load_k = [
        Flatten(
            graph=step_graph,
            input=OffChipLoad(
                underlying=key[i],
                stride=(1, 1),
                out_shape_tiled=(idx[i].shape[0], 1),
                tile_row=1,
                tile_col=head_dim,
                par_dispatch=par_dispatch,
                mock_bf16=mock_bf16,
            ),
            min_rank=1,
            max_rank=2,
        )
        for i in range(par_factor)
    ]

    # output shape: [1, 16, 1] x par_factor (tile: [1,D])
    # (after flatten): [16, 1] x par_factor
    load_v = [
        Flatten(
            graph=step_graph,
            input=OffChipLoad(
                underlying=value[i],
                stride=(1, 1),
                out_shape_tiled=(idx[i].shape[0], 1),
                tile_row=1,
                tile_col=head_dim,
                par_dispatch=par_dispatch,
                mock_bf16=mock_bf16,
            ),
            min_rank=1,
            max_rank=2,
        )
        for i in range(par_factor)
    ]

    # output shape: [1, 16] x par_factor (tile: [1,1])
    # (after flatten): [16] x par_factor
    load_idx = [
        Flatten(
            graph=step_graph,
            input=MetadataGen(
                tensor=idx[i],
            ),
            min_rank=0,
            max_rank=1,
        )
        for i in range(par_factor)
    ]

    # output shape: [1, 16] x par_factor (tile: [1,1])
    # (after flatten): [16] x par_factor
    load_seq_len = [
        Flatten(
            graph=step_graph,
            input=MetadataGen(
                tensor=seq_len[i],
            ),
            min_rank=0,
            max_rank=1,
        )
        for i in range(par_factor)
    ]

    # output shape: [1, 16] x par_factor (tile: [1,1])
    # (after flatten): [16] x par_factor
    load_offset = [
        Flatten(
            graph=step_graph,
            input=MetadataGen(
                tensor=offset[i],
            ),
            min_rank=0,
            max_rank=1,
        )
        for i in range(par_factor)
    ]

    # ------------ Stage 2: Generate control data for parallelization ------------
    # Don't need it for static parallelization

    # ------------ Stage 3: Partition data to parallelize ------------
    broadcast_idx_list = [
        Broadcast(graph=step_graph, input=load_idx[i], num_consumers=5)
        for i in range(par_factor)
    ]
    broadcast_seq_len_list = [
        Broadcast(graph=step_graph, input=load_seq_len[i], num_consumers=6)
        for i in range(par_factor)
    ]
    broadcast_offset_list = [
        Broadcast(graph=step_graph, input=load_offset[i], num_consumers=2)
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
            query=load_q[i],
            key=load_k[i],
            value=load_v[i],
            k_cache_underlying=k_cache,
            v_cache_underlying=v_cache,
            output_underlying=output_underlying,
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


def run_static_coarse_par(
    batch: int,
    cache_row_offset_tiled: int,
    tile_N: int,
    metadata_fifo_depth: int,
    cache_write_back_fifo_depth: int,
    model_config,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output_underlying: torch.Tensor,
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

    par_factor = batch // 16

    # ----------- Create input ------------------
    query = [
        torch.randn(16 * model_config.query_per_kvhead, model_config.head_dim)
        for i in range(par_factor)
    ]
    key = [torch.randn(16, model_config.head_dim) for i in range(par_factor)]
    value = [torch.randn(16, model_config.head_dim) for i in range(par_factor)]

    # Build graph
    step_graph = MultiDiGraph()
    output_list = build_static_coarse_par(
        step_graph=step_graph,
        model_config=model_config,
        query=query,
        key=key,
        value=value,
        k_cache=k_cache,
        v_cache=v_cache,
        output_underlying=output_underlying,
        idx=[torch.arange(i * 16, (i + 1) * 16) for i in range(par_factor)],
        seq_len=[
            seq_len[i * 16 : (i + 1) * 16] for i in range(par_factor)
        ],  # [16] x par_factor
        offset=[
            offset[i * 16 : (i + 1) * 16] for i in range(par_factor)
        ],  # [16] x par_factor
        par_factor=par_factor,
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
        OUTPUT_FILENAME = "static_coarse_par"
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


def run_static_coarse_par_b80(
    batch: int,
    cache_row_offset_tiled: int,
    tile_N: int,
    metadata_fifo_depth: int,
    cache_write_back_fifo_depth: int,
    model_config,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output_underlying: torch.Tensor,
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

    par_factor = 64 // 16

    # ----------- Create input ------------------
    query = []
    for i in range(par_factor):
        if i == 0:
            query.append(
                torch.randn(32 * model_config.query_per_kvhead, model_config.head_dim)
            )
        else:
            query.append(
                torch.randn(16 * model_config.query_per_kvhead, model_config.head_dim)
            )

    key = []
    for i in range(par_factor):
        if i == 0:
            key.append(torch.randn(32, model_config.head_dim))
        else:
            key.append(torch.randn(16, model_config.head_dim))

    value = []
    for i in range(par_factor):
        if i == 0:
            value.append(torch.randn(32, model_config.head_dim))
        else:
            value.append(torch.randn(16, model_config.head_dim))

    idx_list = []
    for i in range(par_factor):
        if i == 0:
            first = torch.arange(i * 16, (i + 1) * 16)
            last = torch.arange(par_factor * 16, (par_factor + 1) * 16)
            idx_list.append(torch.cat([first, last]))
        else:
            idx_list.append(torch.arange(i * 16, (i + 1) * 16))

    seq_len_list = []
    for i in range(par_factor):
        if i == 0:
            first = seq_len[i * 16 : (i + 1) * 16]
            last = seq_len[par_factor * 16 : (par_factor + 1) * 16]
            seq_len_list.append(torch.cat([first, last]))
        else:
            seq_len_list.append(seq_len[i * 16 : (i + 1) * 16])

    offset_list = []
    for i in range(par_factor):
        if i == 0:
            first = offset[i * 16 : (i + 1) * 16]
            last = offset[par_factor * 16 : (par_factor + 1) * 16]
            offset_list.append(torch.cat([first, last]))
        else:
            offset_list.append(offset[i * 16 : (i + 1) * 16])

    # Build graph
    step_graph = MultiDiGraph()
    output_list = build_static_coarse_par(
        step_graph=step_graph,
        model_config=model_config,
        query=query,
        key=key,
        value=value,
        k_cache=k_cache,
        v_cache=v_cache,
        output_underlying=output_underlying,
        idx=idx_list,
        seq_len=seq_len_list,  # [16] x par_factor
        offset=offset_list,  # [16] x par_factor
        par_factor=par_factor,
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
        OUTPUT_FILENAME = "static_coarse_par"
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


def test_static_coarse_par():
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
    maxN = 4160
    tile_N = 32
    cache_row_offset_tiled = maxN // tile_N

    # ====== Input config ======
    batch = 16

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

    assert batch == high - low + 1, "Batch must be equal to high - low + 1"

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
    # k_cache: [batch, maxN, head_dim]
    # v_cache: [batch, maxN, head_dim]
    for i in range(batch):
        k_cache[i, : num_token_list[i]] = torch.randn(
            num_token_list[i], model_config.head_dim
        )
        v_cache[i, : num_token_list[i]] = torch.randn(
            num_token_list[i], model_config.head_dim
        )

    run_static_coarse_par(
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
