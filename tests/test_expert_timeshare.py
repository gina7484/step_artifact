import torch
from networkx import MultiDiGraph
import csv
import numpy as np

from rewrite.broadcast import infer_broadcast
from sim import HBMConfig, SimConfig, simulate

from step_py.utility_ops import *
from step_py.ops import *
from step_py.datatype import DynTile
from step_py.kernels.linear import LinearTileConfig
from step_py.functions import map_accum_fn, map_fn, init_fn, accum_fn

from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from utils.moe import *


def test_random_offchip_load():
    # configuration
    dim = 32
    moe_inter_dim = 128
    num_experts = 8

    tile_size = 32
    num_tile_per_expert = moe_inter_dim // tile_size

    num_selected_experts = 4

    # Expert selection (8 experts)
    routing_expert_selection = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],  # 1
            [1, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 1, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0, 0, 1],  # 7
        ]
    )  # [4,8]

    # Weights for experts
    weights = torch.randn(num_experts * dim, moe_inter_dim)

    # Step graph
    graph = MultiDiGraph()

    control = SelectGen(
        is_multihot=True,
        tensor=routing_expert_selection,
        n=8,
    )  # [1,4]

    flattened_control = Flatten(
        graph=graph,
        input=control,
        min_rank=0,
        max_rank=1,
    )  # [4]

    expert_addr_gen = ExpertAddrGen(
        graph=graph,
        input=flattened_control,
        num_tile_per_expert=num_tile_per_expert,
        expert_addr_base=0,
    )  # [4,4,1]

    random_off_chip_load = RandomOffChipLoad(
        graph=graph,
        underlying=weights,
        raddr=expert_addr_gen,
        tile_row=dim,
        tile_col=tile_size,
        base_addr_byte=0,
        par_dispatch=4,
    )
    # [4,4,1]

    # to mimic the behavior of accum
    accum_weights = Flatten(
        graph=graph,
        input=random_off_chip_load,
        min_rank=0,
        max_rank=1,
    )  # [4,4]

    reshaped_off_chip_load = Reshape(
        graph=graph,
        input=accum_weights,
        chunk_size=num_selected_experts,
        reshape_rank=1,
        write_back_mu=True,
    )  # [1,4,4]

    output = OffChipStore(
        graph=graph,
        input=reshaped_off_chip_load,
        par_dispatch=4,
        store_file_name="output",
    )

    print(output.get_untiled_shape())  # [4*32,128]

    graph = infer_broadcast(graph)

    OUTPUT_FILENAME = "test_random_offchip_load"
    save_graph_format(graph, OUTPUT_FILENAME, ["png"])

    # HBMConfig, SimConfig
    hbm_config = HBMConfig(64, 8, 2, 2, 1, 14)
    sim_config = SimConfig(channel_depth=64, functional_sim=True, mock_bf16=False)

    cycles, duration_ms, duration_s = simulate(
        graph,
        False,  # logging
        hbm_config,
        sim_config,
        "/home/ginasohn/step_tl/graph.pb",
    )

    # Gold generation
    # Define the slice ranges (corrected to get 32 elements each)
    slice_ranges = [
        slice(32, 64),  # [32:64] - indices 32 to 63 (32 elements)
        slice(0, 32),  # [0:32] - indices 0 to 31 (32 elements)
        slice(128, 160),  # [4*32:5*32] - indices 128 to 159 (32 elements)
        slice(224, 256),  # [7*32:8*32] - indices 224 to 255 (32 elements)
    ]

    # Method 1: Using list comprehension and torch.stack
    gathered_slices = [weights[slice_range, :] for slice_range in slice_ranges]
    result_tensor = torch.cat(gathered_slices, dim=0)  # [4*32, 128]

    # Verification
    check_gold_tensor(output.store_file_name, result_tensor.contiguous())


def timeshare_mn_mk_gemm_reshape(
    step_graph: MultiDiGraph,
    model_config,
    batch: int,
    gate_compute_bw: int,
    up_compute_bw: int,
    act_fn_compute_bw: int,
    mult_compute_bw: int,
    down_compute_bw: int,
    weight_scale_compute_bw: int,
    accum_compute_bw: int,
    input_tensor: torch.Tensor,
    expert_multihot: torch.Tensor,
    expert_onehot: torch.Tensor,
    expert_weights: torch.Tensor,
    w_gate_list: torch.Tensor,
    w_up_list: torch.Tensor,
    w_down_list: torch.Tensor,
    tile_N: int,  # M
    tile_F: int,  # Gate & Up (K), Down (N)
    n_par_region: int,
    mock_bf16: bool,
) -> OffChipStore:
    """
    Constructs the STeP graph
    """
    F = model_config.moe_inter_dim
    D = model_config.dim

    # ------------ Stage 1: Load input tensor ------------
    # - tensor shape: [B, D]
    # - stream shape: [1, B, 1] (tile: [1, D])
    in_load = OffChipLoad(
        underlying=input_tensor,
        stride=(
            D // D,
            1,
        ),
        out_shape_tiled=(
            batch,
            1,
        ),
        tile_row=1,
        tile_col=D,
        par_dispatch=4,
        mock_bf16=mock_bf16,
    )

    flatten_in_load = Flatten(
        graph=step_graph,
        input=in_load,
        min_rank=0,
        max_rank=1,
    )

    # ------------ Stage 2: Generate the selection stream ------------
    # - tensor shape: [B, n_routed_experts]
    # - stream shape: [1, B] (tile: Multihot)
    feature_select_gen = SelectGen(
        is_multihot=True, tensor=expert_multihot, n=model_config.n_routed_experts
    )

    # - tensor shape: [B, n_activated_experts, n_routed_experts]
    # - stream shape: [1, B, n_activated_experts] (tile: Multihot)
    weight_select_gen = SelectGen(
        is_multihot=True, tensor=expert_onehot, n=model_config.n_activated_experts
    )

    # ------------ Stage 3: Load the weights for expert weighted sum ------------
    # - tensor shape: [B, n_activated_experts]
    # - stream shape: [1, B, n_activated_experts] (tile: [1, 1])
    weights_load = OffChipLoad(
        underlying=expert_weights,
        stride=(model_config.n_activated_experts, 1),
        out_shape_tiled=(batch, model_config.n_activated_experts),
        tile_row=1,
        tile_col=1,
        par_dispatch=4,
        mock_bf16=mock_bf16,
    )

    # ------------ Stage 4: Partition the input feature stream ------------
    # - input stream shape:   [1, B]
    # - control stream shape: [1, B]
    # - partition_rank: 0
    # - output stream: [Dyn] x n_routed_experts (tile: [1, D])
    unchunked_expert_feature_streams = FlatPartition(
        step_graph,
        flatten_in_load,  # [1, B]
        feature_select_gen,  # [1, B]
        partition_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
        num_consumers=model_config.n_routed_experts,
    )

    # - input stream shape: [Dyn] x n_routed_experts (tile: [1, D])
    # - output stream:
    #   - After Reshape: [(Dyn + tile_N -1) // tile_N, tile_N] x n_routed_experts (tile: [1, D])
    #   - After Accum:   [(Dyn + tile_N -1) // tile_N]         x n_routed_experts (tile: [tile_N, D])

    reshaped_expert_feature_streams = [
        Reshape(
            step_graph,
            (unchunked_expert_feature_streams, i),
            tile_N,
            0,
            write_back_mu=False,
            add_outer_dim=False,
            pad_fn=init_fn.Zero(shape=(1, D), dtype=Float32()),
        )
        for i in range(model_config.n_routed_experts)
    ]

    # In this case, we don't need to specify the outermost 1 is a dynamic dim
    # as it gets flattened with the dyn dim

    expert_feature_streams = [
        Accum(
            step_graph,
            reshaped_expert_feature_streams[i],
            Tile(tile_dtype=Float32(), shape=(tile_N, D)),  # output type
            accum_fn.RetileRow(),
            init_fn.Empty(shape=(0, D), dtype=Float32()),
            1,
            False,
            1024,
        )
        for i in range(model_config.n_routed_experts)
    ]  # [dyn_1 * (Dyn + tile_N -1) // tile_N] of tile_N x D

    # ------------ Stage 5: Reroute input features ------------
    # 5.1: Merge the input features
    # - input stream shape:   [(Dyn + tile_N -1) // tile_N] x n_routed_experts (tile: [tile_N, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N)] (tile: [tile_N, D])
    merged_feature_stream = EagerMerge(
        step_graph,
        expert_feature_streams,
        input_rank=0,
    )

    # 5.2: Parallelize the data and selection
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N)] (tile: [tile_N, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region] x n_par_region (tile: [tile_N, D])
    parallelized_feature_stream = Parallelize(
        graph=step_graph,
        input=merged_feature_stream.data_tuple(),
        parallelize_rank=0,
        num_consumers=n_par_region,  # par_factor
        per_region_input=1,
    )

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N)]                               (dtype: Multihot)
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region] x n_par_region (dtype: Multihot)
    parallelized_selection_stream = Parallelize(
        graph=step_graph,
        input=merged_feature_stream.select_tuple(),
        parallelize_rank=0,
        num_consumers=n_par_region,  # par_factor
        per_region_input=1,
    )

    # The broadcast infer logic gets too complicated if we try to make it also infer cases where 1:M operator is
    # fed into N:1 operator. Therefore, for these cases, we will manually broadcast.
    broadcasted_par_sel_stream = [
        Broadcast(
            graph=step_graph,
            input=(parallelized_selection_stream, i),
            num_consumers=2,
        )
        for i in range(n_par_region)
    ]

    # 5.3: Repeat the input features
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region]              x n_par_region (tile: [tile_N, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] x n_par_region (tile: [tile_N, D])
    repeated_feature_streams = [
        RepeatStatic(
            graph=step_graph,
            input=(parallelized_feature_stream, i),
            repeat_factor=F // tile_F,
        )
        for i in range(n_par_region)
    ]

    # 5.4: Generate addresses for expert loading
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region]                 x n_par_region (tile: [tile_N, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] x n_par_region (tile: [tile_N, D])
    expert_addr_gen = [
        ExpertAddrGen(
            graph=step_graph,
            input=(broadcasted_par_sel_stream[i], 0),
            num_tile_per_expert=F // tile_F,
            expert_addr_base=0,
        )
        for i in range(n_par_region)
    ]

    # ------------ Stage 6: Load up parameters ------------
    # - tensor shape: [D, F]
    # - raddr stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1]
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] (tile: [D, tile_F])
    up_loads = [
        RandomOffChipLoad(
            graph=step_graph,
            raddr=raddr_stream,
            underlying=w_up_list,
            tile_row=D,
            tile_col=tile_F,
            base_addr_byte=0,
            par_dispatch=4,
            mock_bf16=mock_bf16,
        )
        for raddr_stream in expert_addr_gen
    ]

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F]    (tile: [D, tile_F])
    ready_up_loads = [
        Flatten(
            graph=step_graph,
            input=up_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(n_par_region)
    ]

    # ------------ Stage 7: Compute the up features ------------
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, D])
    # - weight stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    up_feature_streams = [
        BinaryMap(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            False,
            up_compute_bw,
        )
        for feature, weight in zip(repeated_feature_streams, ready_up_loads)
    ]

    # ------------ Stage 8: Load gate parameters ------------
    # - tensor shape: [D, F]
    # - raddr stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1]
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] (tile: [D, tile_F])
    gate_loads = [
        RandomOffChipLoad(
            graph=step_graph,
            raddr=raddr_stream,
            underlying=w_gate_list,
            tile_row=D,
            tile_col=tile_F,
            base_addr_byte=0,
            par_dispatch=4,
            mock_bf16=mock_bf16,
        )
        for raddr_stream in expert_addr_gen
    ]

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F]    (tile: [D, tile_F])
    ready_gate_loads = [
        Flatten(
            graph=step_graph,
            input=gate_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(n_par_region)
    ]

    # ------------ Stage 8: Compute the gate features ------------
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, D])
    # - weight stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    pre_act_gate_feature_streams = [
        BinaryMap(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            False,
            gate_compute_bw,
        )
        for feature, weight in zip(repeated_feature_streams, ready_gate_loads)
    ]

    # ------------ Stage 9: Compute the activation ------------
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    gate_feature_streams = [
        UnaryMap(
            graph=step_graph,
            input=feature,
            fn=map_fn.Silu(),
            write_back_mu=False,
            compute_bw=act_fn_compute_bw,
        )
        for feature in pre_act_gate_feature_streams
    ]

    # ------------ Stage 10: Compute the projected features ------------
    # - input1 stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    # - input2 stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    # - output stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    projected_feature_streams = [
        BinaryMap(
            step_graph, up_feature, gate_feature, map_fn.Mul(), False, mult_compute_bw
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]

    # ------------ Stage 11: Load down parameters ------------
    # - tensor shape: [F, D]
    # - raddr stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1]
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] (tile: [tile_F, D])
    down_loads = [
        RandomOffChipLoad(
            graph=step_graph,
            raddr=raddr_stream,
            underlying=w_down_list,
            tile_row=tile_F,
            tile_col=D,
            base_addr_byte=0,
            par_dispatch=4,
            mock_bf16=mock_bf16,
        )
        for raddr_stream in expert_addr_gen
    ]

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_F, D])
    ready_down_loads = [
        Flatten(
            graph=step_graph,
            input=down_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(n_par_region)
    ]

    # ------------ Stage 12: Compute the down features ------------
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_N, tile_F])
    # - weight stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region, F // tile_F] (tile: [tile_F, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region]              (tile: [tile_N, D])
    chunked_down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_accum_fn.Matmul(weight_transposed=False),
            init_fn.Zero(shape=(tile_N, D), dtype=Float32()),
            1,
            False,
            down_compute_bw,
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ]

    # ------------ Additional Stage for Redistributing the experts -----------
    # - input stream shape:  [Σ((Dyn + tile_N -1) // tile_N) / n_par_region] x n_par_region (tile: [tile_N, D])
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N)]                x 1            (tile: [tile_N, D])
    merged_expert_results = EagerMerge(
        graph=step_graph,
        inputs=chunked_down_feature_streams,
        input_rank=0,
    )

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N) / n_par_region] x n_par_region (dtype: Multihot)
    # - control stream shape: [Σ((Dyn + tile_N -1) // tile_N)]                x 1            (dtype: Multihot)
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N)]                x 1            (dtype: Multihot)
    reassembled_expert_idx = FlatReassemble(
        graph=step_graph,
        inputs=[(broadcasted_par_sel_stream[i], 1) for i in range(n_par_region)],
        control=merged_expert_results.select_tuple(),
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(n_par_region)],
        write_back_mu=False,
    )

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N)] x 1                (tile: [tile_N, D])
    # - control stream shape: [Σ((Dyn + tile_N -1) // tile_N)] x 1                (dtype: Multihot)
    # - output stream shape:  [  (Dyn + tile_N -1) // tile_N ] x n_routed_experts (tile: [tile_N, D])
    per_expert_results = FlatPartition(
        graph=step_graph,
        input=merged_expert_results.data_tuple(),
        control=reassembled_expert_idx,
        partition_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
        num_consumers=model_config.n_routed_experts,
    )

    # ------------ Stage 12.5: Partition & Retile outputs for each expert ------------
    # - input stream shape:  [(Dyn + tile_N -1) // tile_N] (tile: [tile_N, D])
    # - output stream shape: [Dyn_retile]                  (tile: [1, D])
    down_feature_streams = [
        RetileStreamify(
            graph=step_graph,
            input=(per_expert_results, i),
            split_row=True,
            filter_mask=True,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # Replace the Dyndim with the actual value
    for partitioned_stream, retiled_stream in zip(
        unchunked_expert_feature_streams.stream_list, down_feature_streams
    ):
        dyn_i = partitioned_stream.shape[0]
        retiled_stream.stream.shape = (dyn_i,)

    # ------------ Stage 13: Partition the scalar weights ------------
    # - input stream shape:   [1, B] (tile: [1, 1])
    # - control stream shape: [1, B]
    # - partition_rank: 0
    # - output stream: [Dyn] x n_routed_experts (tile: [1, 1])

    expert_weight_streams = FlatPartition(
        step_graph,
        weights_load,
        weight_select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
        num_consumers=model_config.n_routed_experts,
    )

    # ------------ Stage 14: Compute the weighted features ------------
    # - input1 stream shape:   [Dyn] (tile: [1, 1])
    # - input2 stream shape:   [Dyn] (tile: [1, D])
    # - output stream shape:   [Dyn] (tile: [1, D])
    weighted_feature_streams = [
        BinaryMap(
            step_graph,
            (expert_weight_streams, i),
            down_feature_streams[i],
            map_fn.Mul(),
            False,
            weight_scale_compute_bw,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 15: Reassemble the weighted features ------------
    # Reassemble
    feature_select_gen_reassemble = SelectGen(
        is_multihot=True, tensor=expert_multihot, n=model_config.n_routed_experts
    )

    reassembled_stream = FlatReassemble(
        step_graph,
        weighted_feature_streams,  # [Dyn] # type: ignore (Cannot infer type of weighted_feature_streams properly)
        feature_select_gen_reassemble,  # [1, N]
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
    )  # [1, N, Ragged]

    # ------------ Stage 16: Accumulate the reassembled features ------------
    accumed_stream = Accum(
        step_graph,
        reassembled_stream,  # [1, N, Ragged]
        Tile(tile_dtype=Float32(), shape=(1, D)),
        accum_fn.Add(),
        init_fn.Zero(shape=(1, D), dtype=Float32()),
        1,
        False,
        accum_compute_bw,
    )  # [1, N] (tile: [1, D])

    # ------------ Stage 17: Store the output ------------
    output = OffChipStore(
        step_graph,
        Reshape(
            graph=step_graph,
            input=accumed_stream,
            chunk_size=1,
            reshape_rank=0,
            write_back_mu=True,
        ),
        par_dispatch=4,
        store_file_name="output",
    )  # [1, N, 1] (tile: [1, D])

    return output


def call_timeshare_mn_mk_gemm_reshape(
    model_config,
    batch: int,
    gate_compute_bw: int,
    up_compute_bw: int,
    act_fn_compute_bw: int,
    mult_compute_bw: int,
    down_compute_bw: int,
    weight_scale_compute_bw: int,
    accum_compute_bw: int,
    input_tensor: torch.Tensor,
    expert_multihot: torch.Tensor,
    expert_onehot: torch.Tensor,
    expert_weights: torch.Tensor,
    w_gate_list: torch.Tensor,
    w_up_list: torch.Tensor,
    w_down_list: torch.Tensor,
    tile_N: int,
    tile_F: int,
    n_par_region: int,
    save_graph: bool,
    simulate_rust: str,
    logging: Optional[str] = None,
    mock_bf16: bool = False,
) -> tuple[StepOps, sympy.Expr, sympy.Expr, int, int]:
    """
    1. Instantiate the graph
    2. Infer Broadcast
    3. Save graph
    4. Calculate off-chip traffic & on-chip requirement
    5. Simulate the graph
    """

    # ------------ 1. Construct the graph ------------
    step_graph = MultiDiGraph()

    output: OffChipStore = timeshare_mn_mk_gemm_reshape(
        step_graph=step_graph,
        model_config=model_config,
        batch=batch,
        gate_compute_bw=gate_compute_bw,
        up_compute_bw=up_compute_bw,
        act_fn_compute_bw=act_fn_compute_bw,
        mult_compute_bw=mult_compute_bw,
        down_compute_bw=down_compute_bw,
        weight_scale_compute_bw=weight_scale_compute_bw,
        accum_compute_bw=accum_compute_bw,
        input_tensor=input_tensor,
        expert_multihot=expert_multihot,
        expert_onehot=expert_onehot,
        expert_weights=expert_weights,
        w_gate_list=w_gate_list,
        w_up_list=w_up_list,
        w_down_list=w_down_list,
        tile_N=tile_N,
        tile_F=tile_F,
        n_par_region=n_par_region,
        mock_bf16=mock_bf16,
    )

    print(f"Output untiled: {output.get_untiled_shape()}")

    # ------------ 2. Infer Broadcast ------------
    step_graph = infer_broadcast(step_graph)

    # ------------ 3. Save graph ------------
    if save_graph:
        OUTPUT_FILENAME = "moe_timeshare_expert_par_gemm"
        save_graph_format(step_graph, OUTPUT_FILENAME, ["svg"])

    # ------------ 4. Calculate off-chip traffic & on-chip requirement ------------
    total_off_chip_traffic = sympy.Integer(0)
    total_on_chip_requirement = sympy.Integer(0)

    # off_chip_traffic_list = {}
    for node_tuple in step_graph.nodes(data=True):
        node, data = node_tuple
        if isinstance(node, StepOps):
            # if node.off_chip_traffic() != 0:
            #     off_chip_traffic_list[
            #         f"{node.__class__.__name__}_{node.instance_id}"
            #     ] = node.off_chip_traffic()
            total_off_chip_traffic = sympy.Add(
                total_off_chip_traffic, node.off_chip_traffic()
            )
            total_on_chip_requirement = sympy.Add(
                total_on_chip_requirement, node.on_chip_requirement()
            )
        else:
            raise ValueError(f"Node {node} in the graph is not a StepOps")

    # for key, value in sorted(off_chip_traffic_list.items()):
    #     print(f"{key}: {value}")

    # ------------ 5. Simulate the graph ------------
    cycles = 0
    duration_ms = 0
    duration_s = 0

    if simulate_rust in ["full", "timing"]:
        hbm_config = HBMConfig(64, 8, 2, 2, 1, 14)
        sim_config = SimConfig(
            channel_depth=2, functional_sim=simulate_rust == "full", mock_bf16=mock_bf16
        )

        if logging is None:
            cycles, duration_ms, duration_s = simulate(
                step_graph,
                False,  # logging
                hbm_config,
                sim_config,
                "/home/ginasohn/step_tl/graph.pb",
            )
        else:
            assert isinstance(logging, str), "Logging must be a string path"
            cycles, duration_ms, duration_s = simulate(
                step_graph,
                True,  # logging
                hbm_config,
                sim_config,
                "/home/ginasohn/step_tl/graph.pb",
                logging,
            )

    return (
        output,
        total_off_chip_traffic,
        total_on_chip_requirement,
        cycles,
        duration_s,
    )


def run_timeshare_mn_mk_gemm_reshape(
    tile_N: int,  # M (The number of requests to chunk for the GEMM in each expert)
    tile_F: int,  # K (The tile size used for the model_config.dim dimension)
    input_tensor,
    expert_indices,
    model_config,
    simulate_rust,  # either "full", "timing", None
    gold_check,
    save_graph: bool,
    n_par_region: int,
    mock_bf16: bool = False,
    logging: Optional[str] = None,
):
    """
    1. Allocate FLOPs
    2. Generate input tensors
    3. Generate expert selection data & routing weights
    4. Generate expert weights
    5. Run the graph
    6. Compare with gold
    """

    B = expert_indices.shape[0]

    # ------------ 1. Allocate FLOPs (Compute Bandwidths) ------------
    GATE_COMPUTE_BW = 4096
    UP_COMPUTE_BW = 4096
    ACT_FN_COMPUTE_BW = 4096
    MULT_COMPUTE_BW = 4096
    DOWN_COMPUTE_BW = 4096
    WEIGHT_SCALE_COMPUTE_BW = 4096
    ACCUM_COMPUTE_BW = 4096

    # ------------ 2. Generate input tensor ------------
    input_tensor = torch.randn(B, model_config.dim)

    # ------------ 3. Generate expert selection data & Routing Weights ------------
    expert_multihot = topk_to_multihot(
        expert_indices, model_config.n_routed_experts
    )  # [B, n_routed_experts]
    expert_onehot = topk_to_onehot(
        expert_indices, model_config.n_routed_experts
    )  # [B, n_activated_experts, n_routed_experts]

    # Expert routing weights
    # Apply softmax to normalize the weights
    expert_weights = torch.softmax(
        torch.randn(B, model_config.n_activated_experts), dim=-1
    )  # [B, n_activated_experts]

    # ------------ 4. Expert Weights (gate, up, down) ------------
    linear_gate_list = [
        torch.nn.Linear(model_config.dim, model_config.moe_inter_dim, bias=False)
        for _ in range(model_config.n_routed_experts)
    ]
    linear_up_list = [
        torch.nn.Linear(model_config.dim, model_config.moe_inter_dim, bias=False)
        for _ in range(model_config.n_routed_experts)
    ]
    linear_down_list = [
        torch.nn.Linear(model_config.moe_inter_dim, model_config.dim, bias=False)
        for _ in range(model_config.n_routed_experts)
    ]

    w_gate_list = [
        linear_gate.weight.T.detach().clone().contiguous()
        for linear_gate in linear_gate_list
    ]
    w_up_list = [
        linear_up.weight.T.detach().clone().contiguous() for linear_up in linear_up_list
    ]
    w_down_list = [
        linear_down.weight.T.detach().clone().contiguous()
        for linear_down in linear_down_list
    ]

    # ------------ 5. Run the graph ------------
    output: OffChipStore
    off_chip_traffic: sympy.Expr
    on_chip_requirement: sympy.Expr

    output, off_chip_traffic, on_chip_requirement, cycles, duration_s = call_timeshare_mn_mk_gemm_reshape(  # type: ignore (Cannot infer type of output properly)
        model_config=model_config,
        batch=B,
        gate_compute_bw=GATE_COMPUTE_BW,
        up_compute_bw=UP_COMPUTE_BW,
        act_fn_compute_bw=ACT_FN_COMPUTE_BW,
        mult_compute_bw=MULT_COMPUTE_BW,
        down_compute_bw=DOWN_COMPUTE_BW,
        weight_scale_compute_bw=WEIGHT_SCALE_COMPUTE_BW,
        accum_compute_bw=ACCUM_COMPUTE_BW,
        input_tensor=input_tensor,
        expert_multihot=expert_multihot,
        expert_onehot=expert_onehot,
        expert_weights=expert_weights,
        w_gate_list=torch.cat(w_gate_list, dim=0),
        w_up_list=torch.cat(w_up_list, dim=0),
        w_down_list=torch.cat(w_down_list, dim=0),
        tile_N=tile_N,
        tile_F=tile_F,
        n_par_region=n_par_region,
        save_graph=save_graph,
        simulate_rust=simulate_rust,
        mock_bf16=mock_bf16,
        logging=logging,
    )

    if simulate_rust and gold_check:
        # Gold calculation
        final_gold = moe_gold_calc(
            input_tensor,
            expert_indices,
            expert_weights,
            linear_gate_list,
            linear_up_list,
            linear_down_list,
        )

        check_gold_tensor(output.store_file_name, final_gold)

    return (off_chip_traffic, on_chip_requirement, cycles, duration_s)


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


def test_timeshare_mn_mk_gemm_reshape():
    mock_bf16 = True
    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    model_config = TinyQwen30b()
    # model_config = Qwen30b()

    tile_Ns = [16]  # For the batch dim (64)
    tile_Fs = [24]  # For the model_config.moe_inter_dim

    # ------------ Expert Indices ------------
    iter = 32
    layer = 12
    expert_selection_file = f"/home/ginasohn/expert_routing/processed_qwen/expr_per_layer/iter_{iter:03d}_layer_{layer:03d}.npz"
    expert_indices_npz = np.load(expert_selection_file)
    expert_indices = torch.from_numpy(
        expert_indices_npz["data"]
    )  # [B, n_activated_experts]

    # expert_counts: [n_routed_experts] (bincount across all batches)
    expert_counts = torch.bincount(
        expert_indices.flatten(), minlength=model_config.n_routed_experts
    )
    print(f"Expert counts: {expert_counts}")

    n_par_region = 32

    # ------------ Input generation -----------
    B = expert_indices.shape[0]

    # Set the random seed
    seed = 5
    torch.manual_seed(seed)

    input_tensor = torch.randn(B, model_config.dim)

    for tile_N in tile_Ns:
        for tile_F in tile_Fs:
            results = []

            off_chip_traffic, on_chip_requirement, cycles, duration_s = (
                run_timeshare_mn_mk_gemm_reshape(
                    tile_N=tile_N,
                    tile_F=tile_F,
                    input_tensor=input_tensor,
                    expert_indices=expert_indices,
                    model_config=model_config,
                    simulate_rust="full",  # "timing",
                    gold_check=True,
                    save_graph=False,
                    n_par_region=n_par_region,
                    mock_bf16=mock_bf16,
                    # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
                )
            )

            # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------

            # ------------ Print the results ------------
            # print(f"Off-chip traffic: {off_chip_traffic}")
            # print(f"On-chip requirement: {on_chip_requirement}")
            print(f"Cycles: {cycles}")
            print(f"Duration: {duration_s} s")
