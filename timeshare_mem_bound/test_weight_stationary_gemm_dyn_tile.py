from networkx import MultiDiGraph
import torch
import csv

from step_py.datatype import DynTile
from step_py.kernels.linear import LinearTileConfig
from step_py.utility_ops import *
from step_py.ops import *
import numpy as np
from sim import SimConfig, simulate, HBMConfig
from step_py.functions import map_accum_fn, map_fn, init_fn, accum_fn
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from utils.moe import *


def ws_tile_mn_mk_gemm_reshape_dyn_tile(
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
    w_gate_list: list[torch.Tensor],
    w_up_list: list[torch.Tensor],
    w_down_list: list[torch.Tensor],
    round_N: int,
    tile_F: int,  # Gate & Up (K), Down (N)
    mock_bf16: bool,
    par_dispatch: int,
) -> Tuple[OffChipStore, int]:
    F = model_config.moe_inter_dim
    D = model_config.dim

    allocated_comp_flops = 0
    unit_expert_on_chip = 0

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
    #   - after reshape: [dyn_1, ((Dyn + round_N -1) // round_N),  round_N] x n_routed_experts (tile: [1, D])
    #   - after flatten: [dyn_1, ((Dyn + round_N -1) // round_N) * round_N] x n_routed_experts (tile: [1, D])
    round_to_16 = [
        Flatten(
            step_graph,
            Reshape(
                step_graph,
                (unchunked_expert_feature_streams, i),
                round_N,
                0,
                write_back_mu=False,
                add_outer_dim=True,
                pad_fn=init_fn.Zero(shape=(1, D), dtype=Float32()),
            ),
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape: [dyn_1, ((Dyn + round_N -1) // round_N) * round_N] x n_routed_experts (tile: [1, D])
    # - output stream: [dyn_1] x n_routed_experts (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    expert_feature_streams = [
        Accum(
            step_graph,
            stream_i,
            DynTile(
                tile_dtype=Float32(), shape=(stream_i.stream.shape[1], D)
            ),  # output type
            accum_fn.RetileRow(),
            init_fn.Empty(shape=(0, D), dtype=Float32()),
            1,
            False,
            1024,
        )
        for stream_i in round_to_16
    ]

    # ------------ Stage 5: Repeat input features ------------
    # - input stream shape:   [dyn_1] x n_routed_experts (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    # - output stream shape:  [dyn_1, F // tile_F] x n_routed_experts (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            expert_feature_streams[i],
            repeat_factor=F // tile_F,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 6: Load up parameters ------------
    # Here, the weight tiles have to be read in a transposed order as we tile the N
    # - tensor shape: [D, F]
    # - output stream shape:  [dyn_1, F // tile_F, 1] (tile: [D, tile_F])
    up_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=w_up_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
        )
        for i in range(model_config.n_routed_experts)
    ]

    unit_expert_on_chip += up_loads[0].on_chip_requirement()

    # - input stream shape:  [1, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape: [1, F // tile_F]    (tile: [D, tile_F])
    ready_up_loads = [
        Flatten(
            graph=step_graph,
            input=up_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 7: Compute the up features ------------
    # - input stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    # - weight stream shape: [dyn_1, F // tile_F] (tile: [D, tile_F])
    # - output stream shape: [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    up_feature_streams = [
        BinaryMap(
            step_graph,
            feature,
            weight,
            map_fn.DynMatmul(weight_transposed=False),
            False,
            up_compute_bw,
        )
        for feature, weight in zip(repeated_feature_streams, ready_up_loads)
    ]
    allocated_comp_flops += up_compute_bw * len(up_feature_streams)

    # ------------ Stage 8: Load gate parameters ------------
    # Here, the weight tiles have to be read in a transposed order as we tile the N
    # - tensor shape: [D, F]
    # - output stream shape: [dyn_1, F // tile_F, 1] (tile: [D, tile_F])
    gate_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=w_gate_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
        )
        for i in range(model_config.n_routed_experts)
    ]

    unit_expert_on_chip += gate_loads[0].on_chip_requirement()

    # - input stream shape:  [1, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape: [1, F // tile_F]    (tile: [D, tile_F])
    ready_gate_loads = [
        Flatten(
            graph=step_graph,
            input=gate_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 8: Compute the gate features ------------
    # - input stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    # - weight stream shape: [dyn_1, F // tile_F] (tile: [D, tile_F])
    # - output stream shape: [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    pre_act_gate_feature_streams = [
        BinaryMap(
            step_graph,
            feature,
            weight,
            map_fn.DynMatmul(weight_transposed=False),
            False,
            gate_compute_bw,
        )
        for feature, weight in zip(repeated_feature_streams, ready_gate_loads)
    ]
    allocated_comp_flops += gate_compute_bw * len(pre_act_gate_feature_streams)

    # ------------ Stage 9: Compute the activation ------------
    # - input stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    # - output stream shape: [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
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
    # - input1 stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    # - input2 stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    # - output stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    projected_feature_streams = [
        BinaryMap(
            step_graph, up_feature, gate_feature, map_fn.Mul(), False, mult_compute_bw
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]
    allocated_comp_flops += mult_compute_bw * len(projected_feature_streams)

    # ------------ Stage 11: Load down parameters ------------
    # Here, the weight tiles don't have to be read in a transposed order as we don't
    # tile the N dimension for the down linear.
    # - tensor shape: [F, D]
    # - per tensor stream shape: [F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape: [dyn_1, F // tile_F, 1] (tile: [tile_F, D])
    down_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=w_down_list[i],
            stride=(D // D, 1),
            out_shape_tiled=(F // tile_F, D // D),
            tile_row=tile_F,
            tile_col=D,
            par_dispatch=par_dispatch,
            mock_bf16=mock_bf16,
        )
        for i in range(model_config.n_routed_experts)
    ]

    unit_expert_on_chip += down_loads[0].on_chip_requirement()

    # - input stream shape:  [dyn_1, F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape: [dyn_1, F // tile_F]    (tile: [tile_F, D])
    ready_down_loads = [
        Flatten(
            graph=step_graph,
            input=down_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 12: Compute the down features ------------
    # - input stream shape:  [dyn_1, F // tile_F] (tile: [((Dyn + round_N -1) // round_N) * round_N, tile_F])
    # - weight stream shape: [dyn_1, F // tile_F] (tile: [tile_F, D])
    # - output stream shape: [dyn_1]              (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    chunked_down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_accum_fn.DynMatmul(weight_transposed=False),
            init_fn.Empty(shape=(0, D), dtype=Float32()),
            1,
            False,
            down_compute_bw,
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ]
    allocated_comp_flops += down_compute_bw * len(chunked_down_feature_streams)

    # ------------ Stage 12.5: Partition & Retile outputs for each expert ------------
    # - input stream shape:  [dyn_1] (tile: [((Dyn + round_N -1) // round_N) * round_N, D])
    # - output stream shape: [Dyn] (tile: [1, D])
    down_feature_streams = [
        RetileStreamify(
            graph=step_graph,
            input=chunked_down_feature_streams[i],
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

    return output, unit_expert_on_chip, allocated_comp_flops


def call_ws_tile_mn_mk_gemm_reshape_dyn_tile(
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
    w_gate_list: list[torch.Tensor],
    w_up_list: list[torch.Tensor],
    w_down_list: list[torch.Tensor],
    round_N: int,
    tile_F: int,
    save_graph: bool,
    simulate_rust: str,
    par_dispatch: int,
    logging: Optional[str] = None,
    mock_bf16: bool = False,
) -> tuple[StepOps, sympy.Expr, sympy.Expr, int, int, int, int]:
    step_graph = MultiDiGraph()

    output: OffChipStore
    allocated_comp_flops: int
    output, unit_expert_on_chip, allocated_comp_flops = (
        ws_tile_mn_mk_gemm_reshape_dyn_tile(
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
            round_N=round_N,
            tile_F=tile_F,
            mock_bf16=mock_bf16,
            par_dispatch=par_dispatch,
        )
    )

    print(f"Output untiled: {output.get_untiled_shape()}")

    step_graph = infer_broadcast(step_graph)

    if save_graph:
        OUTPUT_FILENAME = "moe_expert_par_gemm_reshape"
        save_graph_format(step_graph, OUTPUT_FILENAME, ["svg"])

    total_off_chip_traffic = sympy.Integer(0)
    total_on_chip_requirement = sympy.Integer(0)

    off_chip_traffic_list = {}
    for node_tuple in step_graph.nodes(data=True):
        node, data = node_tuple
        if isinstance(node, StepOps):
            if node.off_chip_traffic() != 0:
                off_chip_traffic_list[
                    f"{node.__class__.__name__}_{node.instance_id}"
                ] = node.off_chip_traffic()
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
        unit_expert_on_chip,
        allocated_comp_flops,
    )


def run_ws_tile_mn_mk_dyn_tile(
    round_N: int,
    tile_F: int,  # K (The tile size used for the model_config.dim dimension)
    input_tensor,
    expert_indices,
    model_config,
    simulate_rust,  # either "full", "timing", None
    gold_check,
    save_graph: bool,
    flops: int,
    flops_for_weighted_sum: int,
    par_dispatch: int,
    mock_bf16: bool = False,
    logging: Optional[str] = None,
):

    B = expert_indices.shape[0]

    # ------------ Tile Configuration ------------
    # For the model_config.moe_inter_dim dimension, we don't tile
    # For Gate & Up linear, we use TileMN. For Down linear, we use TileMK.

    # ------------ Compute Bandwidths ------------
    GATE_COMPUTE_BW = flops
    UP_COMPUTE_BW = flops
    ACT_FN_COMPUTE_BW = flops
    MULT_COMPUTE_BW = flops
    DOWN_COMPUTE_BW = flops
    WEIGHT_SCALE_COMPUTE_BW = flops_for_weighted_sum
    ACCUM_COMPUTE_BW = flops_for_weighted_sum

    # ------------ Input generation ------------
    input_tensor = torch.randn(B, model_config.dim)

    # ------------ Expert Indices ------------
    # [B, n_routed_experts]
    expert_multihot = topk_to_multihot(expert_indices, model_config.n_routed_experts)

    # [B, n_activated_experts, n_routed_experts]
    expert_onehot = topk_to_onehot(expert_indices, model_config.n_routed_experts)

    # ------------ Expert Routed Weights ------------
    # [B, n_activated_experts]
    # Apply softmax to normalize the weights
    expert_weights = torch.softmax(
        torch.randn(B, model_config.n_activated_experts), dim=-1
    )

    # ------------ Expert Weights (gate, up, down) ------------
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

    output: OffChipStore
    off_chip_traffic: sympy.Expr
    on_chip_requirement: sympy.Expr

    output, off_chip_traffic, on_chip_requirement, cycles, duration_s, unit_expert_on_chip, allocated_comp_flops = call_ws_tile_mn_mk_gemm_reshape_dyn_tile(  # type: ignore (Cannot infer type of output properly)
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
        w_gate_list=w_gate_list,
        w_up_list=w_up_list,
        w_down_list=w_down_list,
        round_N=round_N,
        tile_F=tile_F,
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

        check_gold_tensor(output.store_file_name, final_gold)

    return (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        unit_expert_on_chip,
        allocated_comp_flops,
    )
