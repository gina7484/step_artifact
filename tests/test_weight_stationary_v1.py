from networkx import MultiDiGraph
import torch
from step_py.kernels.linear import LinearTileConfig
from step_py.utility_ops import *
from step_py.ops import *
import numpy as np
from sim import simulate, HBMConfig
from step_py.functions import map_fn, init_fn
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from utils.moe import *


def ws_tile_mn_mk_gemv_revet(
    model_config,
    batch: int,
    gate_compute_bw: int,
    up_compute_bw,
    act_fn_compute_bw,
    mult_compute_bw,
    down_compute_bw,
    input_tensor: torch.Tensor,
    expert_multihot: torch.Tensor,
    expert_onehot: torch.Tensor,
    expert_weights: torch.Tensor,
    w_gate_list: list[torch.Tensor],
    w_up_list: list[torch.Tensor],
    w_down_list: list[torch.Tensor],
    tile_F: int,
):
    F = model_config.moe_inter_dim
    D = model_config.dim

    step_graph: MultiDiGraph = MultiDiGraph()

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
        is_multihot=True,
        tensor=expert_multihot,
    )

    # - tensor shape: [B, n_activated_experts, n_routed_experts]
    # - stream shape: [1, B, n_activated_experts] (tile: Multihot)
    weight_select_gen = SelectGen(
        is_multihot=True,
        tensor=expert_onehot,
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
    )

    # ------------ Stage 4: Partition the input feature stream ------------
    # - input stream shape:   [1, B]
    # - control stream shape: [1, B]
    # - partition_rank: 0
    # - output stream: [Dyn] x n_routed_experts (tile: [1, D])
    expert_feature_streams = FlatPartition(
        step_graph,
        flatten_in_load,  # [1, B]
        feature_select_gen,  # [1, B]
        partition_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
        num_consumers=model_config.n_routed_experts,
    )

    # ------------ Stage 5: Repeat input features ------------
    # - input stream shape:   [Dyn] x n_routed_experts
    # - output stream shape:  [Dyn, F // tile_F] x n_routed_experts
    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            (expert_feature_streams, i),
            repeat_factor=F // tile_F,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 6: Load up parameters ------------
    # - tensor shape: [D, F]
    # - ref stream shape: [Dyn]
    # - per tensor stream shape:   [F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    up_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=(expert_feature_streams, i),
            underlying=w_up_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F]    (tile: [D, tile_F])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, D])
    # - weight stream shape:  [Dyn, F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F] (tile: [1, tile_F])
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
    # - ref stream shape: [Dyn]
    # - per tensor stream shape:   [F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    gate_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=(expert_feature_streams, i),
            underlying=w_gate_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F]    (tile: [D, tile_F])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, D])
    # - weight stream shape:  [Dyn, F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F] (tile: [1, tile_F])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - output stream shape:  [Dyn, F // tile_F] (tile: [1, tile_F])
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
    # - input1 stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - input2 stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - output stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    projected_feature_streams = [
        BinaryMap(
            step_graph, up_feature, gate_feature, map_fn.Mul(), False, mult_compute_bw
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]

    # ------------ Stage 11: Load down parameters ------------
    # - tensor shape: [F, D]
    # - ref stream shape: [Dyn]
    # - per tensor stream shape:   [F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [Dyn, F // tile_F, 1] (tile: [tile_F, D])
    down_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=(expert_feature_streams, i),
            underlying=w_down_list[i],
            stride=(D // D, 1),
            out_shape_tiled=(F // tile_F, D // D),
            tile_row=tile_F,
            tile_col=D,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [Dyn, F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [Dyn, F // tile_F]    (tile: [tile_F, D])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - weight stream shape:  [Dyn, F // tile_F] (tile: [tile_F, D])
    # - output stream shape:  [Dyn] (tile: [1, D])
    down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            init_fn.Zero(shape=(1, D), dtype=Float32()),
            1,
            False,
            down_compute_bw,
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ]

    # ------------ Stage 13: Partition the scalar weights ------------
    # - input stream shape:   [1, B]
    # - control stream shape: [1, B]
    # - partition_rank: 0
    # - output stream: [Dyn] x n_routed_experts (tile: [1, D])
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
    # - input1 stream shape:   [Dyn] (tile: [1, D])
    # - input2 stream shape:   [Dyn] (tile: [1, 1])
    # - output stream shape:   [Dyn] (tile: [1, D])
    weighted_feature_streams = [
        BinaryMap(
            step_graph,
            (expert_weight_streams, i),
            down_feature_streams[i],
            map_fn.Mul(),
            False,
            1024,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 15: Reassemble the weighted features ------------
    # - input streams:   [Dyn] x n_routed_experts (tile: [1, D])
    # - control stream: [1, B]
    # - reassemble_rank: 0
    # - output stream: [1, B, n_activated_experts] (tile: [1, D])
    reassembled_stream = FlatReassemble(
        step_graph,
        weighted_feature_streams,  # [Dyn]
        feature_select_gen,  # [1, B]
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
    )

    # ------------ Stage 16: Accumulate the reassembled features ------------
    # - input stream shape:  [1, B, n_activated_experts] (tile: [1, D])
    # - reduction rank: 1
    # - output stream shape: [1, B] (tile: [1, D])
    accumed_stream = Accum(
        step_graph,
        reassembled_stream,
        Tile(tile_dtype=Float32(), shape=(1, D)),
        map_fn.Add(),
        init_fn.Zero(shape=(1, D), dtype=Float32()),
        1,
        False,
        mult_compute_bw,
    )

    # ------------ Stage 17: Store the output ------------
    output = OffChipStore(
        step_graph,
        accumed_stream,
        par_dispatch=4,
        store_file_name="output",
    )  # [1, B]

    step_graph = infer_broadcast(step_graph)
    OUTPUT_FILENAME = "moe_weight_stationary_gemv_step"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])
    return output


def ws_tile_mn_mk_gemv(
    model_config,
    batch: int,
    gate_compute_bw: int,
    up_compute_bw,
    act_fn_compute_bw,
    mult_compute_bw,
    down_compute_bw,
    input_tensor: torch.Tensor,
    expert_multihot: torch.Tensor,
    expert_onehot: torch.Tensor,
    expert_weights: torch.Tensor,
    w_gate_list: list[torch.Tensor],
    w_up_list: list[torch.Tensor],
    w_down_list: list[torch.Tensor],
    tile_F: int,
):
    F = model_config.moe_inter_dim
    D = model_config.dim

    step_graph: MultiDiGraph = MultiDiGraph()

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
        is_multihot=True,
        tensor=expert_multihot,
    )

    # - tensor shape: [B, n_activated_experts, n_routed_experts]
    # - stream shape: [1, B, n_activated_experts] (tile: Multihot)
    weight_select_gen = SelectGen(
        is_multihot=True,
        tensor=expert_onehot,
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
    )

    # ------------ Stage 4: Partition the input feature stream ------------
    # - input stream shape:   [1, B]
    # - control stream shape: [1, B]
    # - partition_rank: 0
    # - output stream: [Dyn] x n_routed_experts (tile: [1, D])
    expert_feature_streams = FlatPartition(
        step_graph,
        flatten_in_load,  # [1, B]
        feature_select_gen,  # [1, B]
        partition_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
        num_consumers=model_config.n_routed_experts,
    )

    # ------------ Stage 5: Repeat input features ------------
    # - input stream shape:   [Dyn] x n_routed_experts
    # - output stream shape:  [Dyn, F // tile_F] x n_routed_experts
    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            (expert_feature_streams, i),
            repeat_factor=F // tile_F,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 6: Load up parameters ------------
    # - tensor shape: [D, F]
    # - ref stream shape: [Dyn]
    # - per tensor stream shape:   [F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    up_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=(expert_feature_streams, i),
            underlying=w_up_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F]    (tile: [D, tile_F])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, D])
    # - weight stream shape:  [Dyn, F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F] (tile: [1, tile_F])
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
    # - ref stream shape: [Dyn]
    # - per tensor stream shape:   [F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    gate_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=(expert_feature_streams, i),
            underlying=w_gate_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [Dyn, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F]    (tile: [D, tile_F])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, D])
    # - weight stream shape:  [Dyn, F // tile_F] (tile: [D, tile_F])
    # - output stream shape:  [Dyn, F // tile_F] (tile: [1, tile_F])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - output stream shape:  [Dyn, F // tile_F] (tile: [1, tile_F])
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
    # - input1 stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - input2 stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - output stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    projected_feature_streams = [
        BinaryMap(
            step_graph, up_feature, gate_feature, map_fn.Mul(), False, mult_compute_bw
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]

    # ------------ Stage 11: Load down parameters ------------
    # - tensor shape: [F, D]
    # - ref stream shape: [Dyn]
    # - per tensor stream shape:   [F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [Dyn, F // tile_F, 1] (tile: [tile_F, D])
    down_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=(expert_feature_streams, i),
            underlying=w_down_list[i],
            stride=(D // D, 1),
            out_shape_tiled=(F // tile_F, D // D),
            tile_row=tile_F,
            tile_col=D,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [Dyn, F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [Dyn, F // tile_F]    (tile: [tile_F, D])
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
    # - input stream shape:   [Dyn, F // tile_F] (tile: [1, tile_F])
    # - weight stream shape:  [Dyn, F // tile_F] (tile: [tile_F, D])
    # - output stream shape:  [Dyn] (tile: [1, D])
    down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            init_fn.Zero(shape=(1, D), dtype=Float32()),
            1,
            False,
            down_compute_bw,
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ]

    # ------------ Stage 13: Partition the scalar weights ------------
    # - input stream shape:   [1, B]
    # - control stream shape: [1, B]
    # - partition_rank: 0
    # - output stream: [Dyn] x n_routed_experts (tile: [1, D])
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
    # - input1 stream shape:   [Dyn] (tile: [1, D])
    # - input2 stream shape:   [Dyn] (tile: [1, 1])
    # - output stream shape:   [Dyn] (tile: [1, D])
    weighted_feature_streams = [
        BinaryMap(
            step_graph,
            (expert_weight_streams, i),
            down_feature_streams[i],
            map_fn.Mul(),
            False,
            1024,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 15: Reassemble the weighted features ------------
    # - input streams:   [Dyn] x n_routed_experts (tile: [1, D])
    # - control stream: [1, B]
    # - reassemble_rank: 0
    # - output stream: [1, B, n_activated_experts] (tile: [1, D])
    reassembled_stream = FlatReassemble(
        step_graph,
        weighted_feature_streams,  # [Dyn]
        feature_select_gen,  # [1, B]
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(model_config.n_routed_experts)],
        write_back_mu=False,
    )

    # ------------ Stage 16: Accumulate the reassembled features ------------
    # - input stream shape:  [1, B, n_activated_experts] (tile: [1, D])
    # - reduction rank: 1
    # - output stream shape: [1, B] (tile: [1, D])
    accumed_stream = Accum(
        step_graph,
        reassembled_stream,
        Tile(tile_dtype=Float32(), shape=(1, D)),
        map_fn.Add(),
        init_fn.Zero(shape=(1, D), dtype=Float32()),
        1,
        False,
        mult_compute_bw,
    )

    # ------------ Stage 17: Store the output ------------
    # - input stream shape:    [1, B]    (tile: [1, D])
    # - promoted stream shape: [1, B, 1] (tile: [1, D])
    # - untiled tensor: [B,D]
    output = OffChipStore(
        step_graph,
        Promote(step_graph, accumed_stream, 0),
        par_dispatch=4,
        store_file_name="output",
    )

    print(f"Output untiled: {output.get_untiled_shape()}")

    step_graph = infer_broadcast(step_graph)

    OUTPUT_FILENAME = "moe_weight_stationary_gemv_step"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )

    return output


@dataclass
class DeepSeekV316B:
    n_routed_experts = 64
    n_activated_experts = 6
    dim = 2048
    moe_inter_dim = 1408


@dataclass
class SmallerDeepSeekV3:
    n_routed_experts = 64
    n_activated_experts = 6
    dim = 64
    moe_inter_dim = 128


@dataclass
class SmallerMixtral:
    n_routed_experts = 8
    n_activated_experts = 2
    dim = 64
    moe_inter_dim = 128


def test_deepseekv3_ws_tile_mn_mk():
    # ------------ Model Configuration ------------
    model_config = SmallerMixtral()

    # ------------ Batch Size ------------
    B = 64

    # ------------ Compute Bandwidths ------------
    GATE_COMPUTE_BW = 1022
    UP_COMPUTE_BW = 1022
    ACT_FN_COMPUTE_BW = 1022
    MULT_COMPUTE_BW = 1022
    DOWN_COMPUTE_BW = 1022

    # ------------ Input generation ------------
    input_tensor = torch.randn(B, model_config.dim)

    # ------------ Expert Indices ------------
    expert_indices = torch.topk(
        torch.randn(B, model_config.n_routed_experts),
        model_config.n_activated_experts,
        dim=-1,
    )[1]

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

    output: OffChipStore = ws_tile_mn_mk_gemv(
        model_config=model_config,
        batch=B,
        gate_compute_bw=GATE_COMPUTE_BW,
        up_compute_bw=UP_COMPUTE_BW,
        act_fn_compute_bw=ACT_FN_COMPUTE_BW,
        mult_compute_bw=MULT_COMPUTE_BW,
        down_compute_bw=DOWN_COMPUTE_BW,
        input_tensor=input_tensor,
        expert_multihot=expert_multihot,
        expert_onehot=expert_onehot,
        expert_weights=expert_weights,
        w_gate_list=w_gate_list,
        w_up_list=w_up_list,
        w_down_list=w_down_list,
        tile_F=16,
    )

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
