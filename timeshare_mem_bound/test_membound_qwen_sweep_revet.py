from re import T
import time
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

from timeshare_mem_bound.test_weight_stationary_gemm_revet import (
    run_ws_tile_mn_mk_revet,
)


def timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(
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
    par_dispatch: int,
) -> Tuple[OffChipStore, int, int, int]:
    """
    Constructs the STeP graph
    """
    allocated_comp_flops = 0
    per_expert_accum_unit_on_chip = 0
    expert_region_unit_on_chip = 0

    assert model_config.n_routed_experts % n_par_region == 0
    expert_per_region = model_config.n_routed_experts // n_par_region

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
        par_dispatch=par_dispatch,
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
        par_dispatch=par_dispatch,
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
    #   - After Reshape: [dyn_1, (Dyn + tile_N -1) // tile_N, tile_N] x n_routed_experts (tile: [1, D])
    #   - After Flatten: [(Dyn + tile_N -1) // tile_N, tile_N] x n_routed_experts (tile: [1, D])
    #   - After Accum:   [(Dyn + tile_N -1) // tile_N]         x n_routed_experts (tile: [tile_N, D])

    reshaped_expert_feature_streams = [
        Reshape(
            step_graph,
            (unchunked_expert_feature_streams, i),
            tile_N,
            0,
            write_back_mu=False,
            add_outer_dim=True,
            pad_fn=init_fn.Zero(shape=(1, D), dtype=Float32()),
        )
        for i in range(model_config.n_routed_experts)
    ]

    # In this case, we don't need to specify the outermost 1 is a dynamic dim
    # as it gets flattened with the dyn dim

    expert_feature_streams = [
        Accum(
            step_graph,
            Flatten(
                step_graph,
                reshaped_expert_feature_streams[i],
                min_rank=1,
                max_rank=2,
            ),
            Tile(tile_dtype=Float32(), shape=(tile_N, D)),  # output type
            accum_fn.RetileRow(),
            init_fn.Empty(shape=(0, D), dtype=Float32()),
            1,
            False,
            accum_compute_bw,
        )
        for i in range(model_config.n_routed_experts)
    ]  # [dyn_1 * (Dyn + tile_N -1) // tile_N] of tile_N x D

    per_expert_accum_unit_on_chip += expert_feature_streams[0].on_chip_requirement()

    # ------------ Stage 5: Reroute input features ------------
    # 5.1: Merge the input features
    # - input stream shape:         [(Dyn + tile_N -1) // tile_N] x n_routed_experts (tile: [tile_N, D])
    # - output stream shape: (data) [Σ((Dyn + tile_N -1) // tile_N)] x n_par_region (tile: [tile_N, D])
    #                        (sel)  [Σ((Dyn + tile_N -1) // tile_N)] x n_par_region (dtype: Multihot)
    merged_feature_streams = [
        EagerMerge(
            step_graph,
            expert_feature_streams[i * expert_per_region : (i + 1) * expert_per_region],
            input_rank=0,
        )
        for i in range(n_par_region)
    ]

    # 5.2: Repeat the input features
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N)]              x n_par_region (tile: [tile_N, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] x n_par_region (tile: [tile_N, D])
    repeated_feature_streams = [
        RepeatStatic(
            graph=step_graph,
            input=merged_feature_streams[i].data_tuple(),
            repeat_factor=F // tile_F,
        )
        for i in range(n_par_region)
    ]

    expert_region_unit_on_chip += repeated_feature_streams[0].on_chip_requirement()

    # 5.3: Generate addresses for expert loading
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N)]                 x n_par_region (dtype: Multihot)
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] x n_par_region (dtype: u64)
    expert_addr_gen = [
        ExpertAddrGen(
            graph=step_graph,
            input=merged_feature_streams[i].select_tuple(),
            num_tile_per_expert=F // tile_F,
            expert_addr_base=0,
        )
        for i in range(n_par_region)
    ]

    # ------------ Stage 6: Load up parameters ------------
    # - tensor shape: [D, F]
    # - raddr stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1]
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] (tile: [D, tile_F])
    up_loads = [
        RandomOffChipLoad(
            graph=step_graph,
            raddr=expert_addr_gen[i],
            underlying=w_up_list[i],
            tile_row=D,
            tile_col=tile_F,
            base_addr_byte=0,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
        )
        for i in range(n_par_region)
    ]

    expert_region_unit_on_chip += up_loads[0].on_chip_requirement()

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F]    (tile: [D, tile_F])
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
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, D])
    # - weight stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
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
    allocated_comp_flops += up_compute_bw * len(up_feature_streams)

    # ------------ Stage 8: Load gate parameters ------------
    # - tensor shape: [D, F]
    # - raddr stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1]
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] (tile: [D, tile_F])
    gate_loads = [
        RandomOffChipLoad(
            graph=step_graph,
            raddr=expert_addr_gen[i],
            underlying=w_gate_list[i],
            tile_row=D,
            tile_col=tile_F,
            base_addr_byte=0,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
        )
        for i in range(n_par_region)
    ]
    expert_region_unit_on_chip += gate_loads[0].on_chip_requirement()

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F]    (tile: [D, tile_F])
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
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, D])
    # - weight stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
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
    allocated_comp_flops += gate_compute_bw * len(pre_act_gate_feature_streams)

    # ------------ Stage 9: Compute the activation ------------
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
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
    allocated_comp_flops += act_fn_compute_bw * len(gate_feature_streams)

    # ------------ Stage 10: Compute the projected features ------------
    # - input1 stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    # - input2 stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    # - output stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    projected_feature_streams = [
        BinaryMap(
            step_graph, up_feature, gate_feature, map_fn.Mul(), False, mult_compute_bw
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]
    allocated_comp_flops += mult_compute_bw * len(projected_feature_streams)

    # ------------ Stage 11: Load down parameters ------------
    # - tensor shape: [F, D]
    # - raddr stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1]
    # - output stream shape: [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] (tile: [tile_F, D])
    down_loads = [
        RandomOffChipLoad(
            graph=step_graph,
            raddr=expert_addr_gen[i],
            underlying=w_down_list[i],
            tile_row=tile_F,
            tile_col=D,
            base_addr_byte=0,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
        )
        for i in range(n_par_region)
    ]

    expert_region_unit_on_chip += down_loads[0].on_chip_requirement()

    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_F, D])
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
    # - input stream shape:   [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    # - weight stream shape:  [Σ((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_F, D])
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N)]              (tile: [tile_N, D])
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
    allocated_comp_flops += down_compute_bw * len(chunked_down_feature_streams)
    expert_region_unit_on_chip += chunked_down_feature_streams[0].on_chip_requirement()

    # ------------ Additional Stage for Redistributing the experts -----------
    # - output stream shape:  [Σ((Dyn + tile_N -1) // tile_N)] x n_par_region (tile: [tile_N, D])
    # - control stream shape: [Σ((Dyn + tile_N -1) // tile_N)] x n_par_region (dtype: Multihot)
    # - output stream shape:  [  (Dyn + tile_N -1) // tile_N ] x n_routed_experts (tile: [tile_N, D])
    per_expert_results = [
        FlatPartition(
            graph=step_graph,
            input=chunked_down_feature_streams[i],
            control=merged_feature_streams[i].select_tuple(),
            partition_rank=0,
            switch_cycles=[1 for _ in range(expert_per_region)],
            write_back_mu=False,
            num_consumers=expert_per_region,
        )
        for i in range(n_par_region)
    ]

    # ------------ Stage 12.5: Partition & Retile outputs for each expert ------------
    # - input stream shape:  [(Dyn + tile_N -1) // tile_N] (tile: [tile_N, D])
    # - output stream shape: [Dyn_retile]                  (tile: [1, D])
    down_feature_streams = [
        RetileStreamify(
            graph=step_graph,
            input=(per_expert_results[i // expert_per_region], i % expert_per_region),
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
    allocated_comp_flops += weight_scale_compute_bw * len(weighted_feature_streams)

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

    allocated_comp_flops += accum_compute_bw

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
        par_dispatch=par_dispatch,
        store_file_name="output",
    )  # [1, N, 1] (tile: [1, D])

    return (
        output,
        per_expert_accum_unit_on_chip,
        expert_region_unit_on_chip,
        allocated_comp_flops,
    )


def call_timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(
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
    par_dispatch: int,
    logging: Optional[str] = None,
    mock_bf16: bool = False,
) -> tuple[StepOps, sympy.Expr, sympy.Expr, int, int, int, int, int]:
    """
    1. Instantiate the graph
    2. Infer Broadcast
    3. Save graph
    4. Calculate off-chip traffic & on-chip requirement
    5. Simulate the graph
    """

    # ------------ 1. Construct the graph ------------
    step_graph = MultiDiGraph()

    output: OffChipStore
    allocated_comp_flops: int
    (
        output,
        per_expert_accum_unit_on_chip,
        expert_region_unit_on_chip,
        allocated_comp_flops,
    ) = timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(
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
        par_dispatch=par_dispatch,
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
        hbm_config = HBMConfig(64, 32, 2, 2, 1, 14)
        sim_config = SimConfig(
            channel_depth=2, functional_sim=simulate_rust == "full", mock_bf16=mock_bf16
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

    return (
        output,
        total_off_chip_traffic,
        total_on_chip_requirement,
        cycles,
        duration_s,
        per_expert_accum_unit_on_chip,
        expert_region_unit_on_chip,
        allocated_comp_flops,
    )


def run_timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(
    tile_N: int,  # M (The number of requests to chunk for the GEMM in each expert)
    tile_F: int,  # K (The tile size used for the model_config.dim dimension)
    input_tensor,
    expert_indices,
    model_config,
    simulate_rust,  # either "full", "timing", None
    gold_check,
    save_graph: bool,
    n_par_region: int,
    flops: int,
    flops_for_weighted_sum: int,
    par_dispatch: int,
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
    GATE_COMPUTE_BW = flops
    UP_COMPUTE_BW = flops
    ACT_FN_COMPUTE_BW = flops
    MULT_COMPUTE_BW = flops
    DOWN_COMPUTE_BW = flops
    WEIGHT_SCALE_COMPUTE_BW = flops_for_weighted_sum
    ACCUM_COMPUTE_BW = flops_for_weighted_sum

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

    # # When debugging value mismatch, initializing weights to 1 helps
    # for layer in linear_gate_list:
    #     torch.nn.init.ones_(layer.weight)

    # # Same for the other layer lists
    # for layer in linear_up_list:
    #     torch.nn.init.ones_(layer.weight)

    # for layer in linear_down_list:
    #     torch.nn.init.ones_(layer.weight)

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

    expert_per_region = model_config.n_routed_experts // n_par_region

    (
        output,
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        per_expert_accum_unit_on_chip,
        expert_region_unit_on_chip,
        allocated_comp_flops,
    ) = call_timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(  # type: ignore (Cannot infer type of output properly)
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
        w_gate_list=[
            torch.cat(
                w_gate_list[i * expert_per_region : (i + 1) * expert_per_region], dim=0
            )
            for i in range(n_par_region)
        ],
        w_up_list=[
            torch.cat(
                w_up_list[i * expert_per_region : (i + 1) * expert_per_region], dim=0
            )
            for i in range(n_par_region)
        ],
        w_down_list=[
            torch.cat(
                w_down_list[i * expert_per_region : (i + 1) * expert_per_region], dim=0
            )
            for i in range(n_par_region)
        ],
        tile_N=tile_N,
        tile_F=tile_F,
        n_par_region=n_par_region,
        save_graph=save_graph,
        simulate_rust=simulate_rust,
        par_dispatch=par_dispatch,
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

        check_gold_tensor(
            output.store_file_name, final_gold.detach().clone().contiguous()
        )

    return (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        per_expert_accum_unit_on_chip,
        expert_region_unit_on_chip,
        allocated_comp_flops,
    )


@dataclass
class SmallerQwen30b:  # 16x scaled down version for each dimension
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 128  # 2048 // 16
    moe_inter_dim = 128  # 768 // 16


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
    par_dispatch = 1
    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    # model_config = TinyQwen30b()
    model_config = Qwen30b()

    n_par_region_list = [64, 32, 16, 8, 4, 2]
    tile_Ns = [32]  # For the batch dim (64)
    tile_Fs = [48]  # For the model_config.moe_inter_dim

    base_flops = 1024  # unit flop for 128 case (no time sharing)
    flops_for_weighted_sum = 1024  # fixed among sweep

    # ------------ Expert Indices ------------
    iter = 32
    layer = 12
    expert_selection_file = f"expert_routing/processed_qwen/expr_per_layer/iter_{iter:03d}_layer_{layer:03d}.npz"
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

    results = []
    for n_par_region in n_par_region_list:
        unit_flops = base_flops
        for tile_N in tile_Ns:
            for tile_F in tile_Fs:

                (
                    off_chip_traffic,
                    on_chip_requirement,
                    cycles,
                    duration_s,
                    per_expert_accum_unit_on_chip,
                    expert_region_unit_on_chip,
                    allocated_comp_flops,
                ) = run_timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(
                    tile_N=tile_N,
                    tile_F=tile_F,
                    input_tensor=input_tensor,
                    expert_indices=expert_indices,
                    model_config=model_config,
                    simulate_rust="timing",
                    gold_check=False,
                    save_graph=False,
                    n_par_region=n_par_region,
                    flops=unit_flops,
                    flops_for_weighted_sum=flops_for_weighted_sum,
                    par_dispatch=par_dispatch,
                    mock_bf16=mock_bf16,
                    # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
                )

                # ------------ Total FLOPs ------------------
                num_tiles = [
                    (routed_toks + tile_N - 1) // tile_N
                    for routed_toks in expert_counts.tolist()
                ]
                after_pad_batch_dim = [
                    num_tiles_i * tile_N for num_tiles_i in num_tiles
                ]
                alg_flops = sum(
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

                # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
                free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

                sub_dict = {
                    symbol: value
                    for symbol, value in zip(free_symbols, expert_counts.tolist())
                }
                # print(f"off_chip_traffic: {off_chip_traffic}")

                off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

                # --------------- On-chip requirement ---------------
                on_chip_requirement_val = (
                    on_chip_requirement
                    - expert_counts.tolist().count(0) * per_expert_accum_unit_on_chip
                )
                # subtract the on-chip requirement for unselected experts
                token_per_region = []

                # Iterate through the list in steps of n
                for i in range(0, len(expert_counts.tolist()), 128 // n_par_region):
                    # Sum the next n elements (or remaining elements if less than n)
                    chunk_sum = sum(expert_counts.tolist()[i : i + 128 // n_par_region])
                    token_per_region.append(chunk_sum)

                print(f"Unselected region: {token_per_region.count(0)}")
                on_chip_requirement_val = (
                    on_chip_requirement_val
                    - token_per_region.count(0) * expert_region_unit_on_chip
                )

                # ------------ Print the results ------------
                print(f"Off-chip traffic (KB): {round(off_chip_traffic_val / 1024, 2)}")
                print(
                    f"On-chip requirement (KB): {round(on_chip_requirement_val / 1024, 2)}"
                )
                print(f"Cycles: {cycles}")
                print(f"Duration: {duration_s} s")

                dict_to_append = {
                    "batch": B,
                    "n_par_region": n_par_region,
                    "n_expert_per_region": model_config.n_routed_experts
                    // n_par_region,
                    "unit_flops": unit_flops,
                    "tile_N": tile_N,
                    "tile_F": tile_F,
                    "flops": alg_flops,  # FLOP of the algorithm itself
                    "cycles": cycles,
                    "duration_s": duration_s,
                    "off_chip_traffic_bytes": off_chip_traffic_val,
                    "on_chip_requirement_bytes": on_chip_requirement_val,
                    "allocated_flops": allocated_comp_flops,
                    "utilization(%)": round(
                        (alg_flops / cycles) / allocated_comp_flops * 100, 2
                    ),
                }
                print(dict_to_append)
                results.append(dict_to_append)

    out_file = (
        f"timeshare_mem_bound/"
        + f"qwen_{model_config.dim}_{model_config.moe_inter_dim}_"
        + f"iter{iter:03d}_layer_{layer:03d}_n{tile_N}_f{tile_F}_"
        + f"timeshare_membound_revet_"
        + f"{time.strftime("%d%H%M%S")}.csv"
    )
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "batch",
                "n_par_region",
                "n_expert_per_region",
                "unit_flops",
                "tile_N",
                "tile_F",
                "flops",
                "cycles",
                "duration_s",
                "off_chip_traffic_bytes",
                "on_chip_requirement_bytes",
                "allocated_flops",
                "utilization(%)",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in results:
                writer.writerow(result)

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def test_baseline_for_membound_timeshare():
    mock_bf16 = True
    par_dispatch = 1
    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    # model_config = TinyQwen30b()
    model_config = Qwen30b()

    tile_Ns = [32]  # For the batch dim (64)
    tile_Fs = [48]  # For the model_config.moe_inter_dim

    base_flops = 1024  # unit flop for 128 case (no time sharing)
    flops_for_weighted_sum = 1024  # fixed among sweep

    # ------------ Expert Indices ------------
    iter = 32
    layer = 12
    expert_selection_file = f"expert_routing/processed_qwen/expr_per_layer/iter_{iter:03d}_layer_{layer:03d}.npz"
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

    unit_flops = base_flops
    results = []
    for tile_N in tile_Ns:
        for tile_F in tile_Fs:

            (
                off_chip_traffic,
                on_chip_requirement,
                cycles,
                duration_s,
                unit_expert_on_chip,
                allocated_comp_flops,
            ) = run_ws_tile_mn_mk_revet(
                tile_N=tile_N,
                tile_F=tile_F,
                input_tensor=input_tensor,
                expert_indices=expert_indices,
                model_config=model_config,
                simulate_rust="timing",
                gold_check=False,
                save_graph=False,
                flops=unit_flops,
                flops_for_weighted_sum=flops_for_weighted_sum,
                par_dispatch=par_dispatch,
                mock_bf16=mock_bf16,
                # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
            )

            # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
            num_tiles = [
                (routed_toks + tile_N - 1) // tile_N
                for routed_toks in expert_counts.tolist()
            ]
            after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

            alg_flops = sum(
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

            # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
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

            # ------------ Print the results ------------
            dict_to_append = {
                "batch": B,
                "n_par_region": 128,
                "n_expert_per_region": 1,
                "unit_flops": unit_flops,
                "tile_N": tile_N,
                "tile_F": tile_F,
                "flops": alg_flops,
                "cycles": cycles,
                "duration_s": duration_s,
                "off_chip_traffic_bytes": off_chip_traffic_val,
                "on_chip_requirement_bytes": on_chip_requirement_val,
                "allocated_flops": allocated_comp_flops,
                "utilization(%)": round(
                    (alg_flops / cycles) / allocated_comp_flops * 100, 2
                ),
            }
            print(dict_to_append)
            results.append(dict_to_append)

    out_file = (
        f"timeshare_mem_bound/"
        + f"qwen_{model_config.dim}_{model_config.moe_inter_dim}_"
        + f"iter{iter:03d}_layer_{layer:03d}_n{tile_N}_f{tile_F}_"
        + f"timeshare_membound_baseline_revet_"
        + f"{time.strftime("%d%H%M%S")}.csv"
    )
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "batch",
                "n_par_region",
                "n_expert_per_region",
                "unit_flops",
                "tile_N",
                "tile_F",
                "flops",
                "cycles",
                "duration_s",
                "off_chip_traffic_bytes",
                "on_chip_requirement_bytes",
                "allocated_flops",
                "utilization(%)",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for result in results:
                writer.writerow(result)

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

def test_static_tile():
    fig_8_a = {
        "performance(cycles)": [],
        "compute_util(%)": [],
        "performance_overhead(%)": [],
    }

    fig_9_a = {
        "performance(cycles)": [],
        "allocated_comp(flops/cycle)": [],
        "on_chip_mem(KB)": [],
    }
    
    fig_9_b = {
        "performance(cycles)": [],
        "off_chip_bandwidth_utilization(%)": [],
    }


    mock_bf16 = True
    par_dispatch = 1
    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    model_config = Qwen30b()

    tile_N =32
    tile_F = 48 if isinstance(model_config, Qwen30b) else 32

    base_flops = 1024  # unit flop for 128 case (no time sharing)
    flops_for_weighted_sum = 1024  # fixed among sweep


    # ------------ Expert Indices ------------
    iter = 32
    layer = 12
    expert_selection_file = f"./dyn_tiling/expert_routing/qwen_b64/iter_{iter:03d}_layer_{layer:03d}.npz"
    expert_indices_npz = np.load(expert_selection_file)
    expert_indices = torch.from_numpy(
        expert_indices_npz["data"]
    )  # [B, n_activated_experts]

    # expert_counts: [n_routed_experts] (bincount across all batches)
    expert_counts = torch.bincount(
        expert_indices.flatten(), minlength=model_config.n_routed_experts
    )
    

    # ------------ Input generation -----------
    B = expert_indices.shape[0]

    # Set the random seed
    seed = 5
    torch.manual_seed(seed)

    input_tensor = torch.randn(B, model_config.dim)

    
    #################### Basecase (parallel regions = 128) ####################
    basecase_cycles = 0

    unit_flops = base_flops
    (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        unit_expert_on_chip,
        allocated_comp_flops,
    ) = run_ws_tile_mn_mk_revet(
        tile_N=tile_N,
        tile_F=tile_F,
        input_tensor=input_tensor,
        expert_indices=expert_indices,
        model_config=model_config,
        simulate_rust="timing",
        gold_check=False,
        save_graph=False,
        flops=unit_flops,
        flops_for_weighted_sum=flops_for_weighted_sum,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
        # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
    )

    # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
    num_tiles = [
        (routed_toks + tile_N - 1) // tile_N
        for routed_toks in expert_counts.tolist()
    ]
    after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

    alg_flops = sum(
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

    # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
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

    # ------------ Print the results ------------
    off_chip_traffic_val = int(off_chip_traffic_val)
    on_chip_requirement_val = int(on_chip_requirement_val)

    fig_8_a["performance(cycles)"].append(cycles)
    fig_8_a["compute_util(%)"].append(round(
        (alg_flops / cycles) / allocated_comp_flops * 100, 2
    ))
    fig_8_a["performance_overhead(%)"].append(0.0)

    fig_9_a["performance(cycles)"].append(cycles)
    fig_9_a["allocated_comp(flops/cycle)"].append(
        allocated_comp_flops
    )
    fig_9_a["on_chip_mem(KB)"].append(
        round(on_chip_requirement_val / 1000, 2)
    )

    mem_bw = 1024  # (bytes/cycle)
    fig_9_b["performance(cycles)"].append(cycles)
    fig_9_b["off_chip_bandwidth_utilization(%)"].append(
        round((off_chip_traffic_val/cycles)/mem_bw*100,2)
    )

    basecase_cycles = cycles

    #################### Timeshare cases ####################
    n_par_region_list = [64, 32, 16, 8, 4]
    for n_par_region in n_par_region_list:
        unit_flops = base_flops

        (
            off_chip_traffic,
            on_chip_requirement,
            cycles,
            duration_s,
            per_expert_accum_unit_on_chip,
            expert_region_unit_on_chip,
            allocated_comp_flops,
        ) = run_timeshare_mn_mk_gemm_reshape_fixed_fanout_revet(
            tile_N=tile_N,
            tile_F=tile_F,
            input_tensor=input_tensor,
            expert_indices=expert_indices,
            model_config=model_config,
            simulate_rust="timing",
            gold_check=False,
            save_graph=False,
            n_par_region=n_par_region,
            flops=unit_flops,
            flops_for_weighted_sum=flops_for_weighted_sum,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
            # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
        )

        # ------------ Total FLOPs ------------------
        num_tiles = [
            (routed_toks + tile_N - 1) // tile_N
            for routed_toks in expert_counts.tolist()
        ]
        after_pad_batch_dim = [
            num_tiles_i * tile_N for num_tiles_i in num_tiles
        ]
        alg_flops = sum(
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

        # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
        free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

        sub_dict = {
            symbol: value
            for symbol, value in zip(free_symbols, expert_counts.tolist())
        }
        # print(f"off_chip_traffic: {off_chip_traffic}")

        off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

        # --------------- On-chip requirement ---------------
        on_chip_requirement_val = (
            on_chip_requirement
            - expert_counts.tolist().count(0) * per_expert_accum_unit_on_chip
        )
        # subtract the on-chip requirement for unselected experts
        token_per_region = []

        # Iterate through the list in steps of n
        for i in range(0, len(expert_counts.tolist()), 128 // n_par_region):
            # Sum the next n elements (or remaining elements if less than n)
            chunk_sum = sum(expert_counts.tolist()[i : i + 128 // n_par_region])
            token_per_region.append(chunk_sum)

        print(f"Unselected region: {token_per_region.count(0)}")
        on_chip_requirement_val = (
            on_chip_requirement_val
            - token_per_region.count(0) * expert_region_unit_on_chip
        )

        # ------------ Print the results ------------
        off_chip_traffic_val = int(off_chip_traffic_val)
        on_chip_requirement_val = int(on_chip_requirement_val)

        fig_8_a["performance(cycles)"].append(cycles)
        fig_8_a["compute_util(%)"].append(round(
            (alg_flops / cycles) / allocated_comp_flops * 100, 2
        ))
        fig_8_a["performance_overhead(%)"].append(int(cycles / basecase_cycles * 100) - 100)

        fig_9_a["performance(cycles)"].append(cycles)
        fig_9_a["allocated_comp(flops/cycle)"].append(
            allocated_comp_flops
        )
        fig_9_a["on_chip_mem(KB)"].append(
            round(on_chip_requirement_val / 1000, 2)
        )

        mem_bw = 1024  # (bytes/cycle)
        fig_9_b["performance(cycles)"].append(cycles)
        fig_9_b["off_chip_bandwidth_utilization(%)"].append(
        round((off_chip_traffic_val/cycles)/mem_bw*100,2)
        )
    


    out_file = f"./timeshare_mem_bound/fig_8_a.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                "Parallel_regions(experts_per_region)",
                "128(1)",
                "64(2)",
                "32(4)",
                "16(8)",
                "8(16)",
                "4(32)",
            ])
            writer.writerow(
                ["performance(cycles)"] + fig_8_a["performance(cycles)"]
            )
            writer.writerow(
                ["compute_util(%)"] + fig_8_a["compute_util(%)"]
            )
            writer.writerow(
                ["performance_overhead(%)"] + fig_8_a["performance_overhead(%)"]
            )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
     



    out_file = f"./timeshare_mem_bound/fig_9_a.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                "Parallel_regions(experts_per_region)",
                "128(1)",
                "64(2)",
                "32(4)",
                "16(8)",
                "8(16)",
                "4(32)",
            ])
            writer.writerow(
                ["performance(cycles)"] + fig_9_a["performance(cycles)"]
            )
            writer.writerow(
                ["allocated_comp(flops/cycle)"] + fig_9_a["allocated_comp(flops/cycle)"]
            )
            writer.writerow(
                ["on_chip_mem(KB)"] + fig_9_a["on_chip_mem(KB)"]
            )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        


    out_file = f"./timeshare_mem_bound/fig_9_b.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                "Parallel_regions(experts_per_region)",
                "128(1)",
                "64(2)",
                "32(4)",
                "16(8)",
                "8(16)",
                "4(32)",
            ])
            writer.writerow(
                ["performance(cycles)"] + fig_9_b["performance(cycles)"]
            )
            writer.writerow(
                ["off_chip_bandwidth_utilization(%)"] + fig_9_b["off_chip_bandwidth_utilization(%)"]
            )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        