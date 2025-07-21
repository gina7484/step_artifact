from networkx import MultiDiGraph
import torch
from step_py.datatype import DynTile
from step_py.kernels.linear import LinearTileConfig
from step_py.utility_ops import *
from step_py.ops import *
import numpy as np
import csv
from sim import SimConfig, simulate, HBMConfig
from step_py.functions import map_accum_fn, map_fn, init_fn, accum_fn
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from utils.moe import *


def hybrid_moe_gemm(
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
    tile_N: int,  # M
    tile_F: int,  # Gate & Up (K), Down (N)
    group_size: int,  # the number of experts that share the same area
) -> OffChipStore:
    assert model_config.n_routed_experts % group_size == 0

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
    expert_feature_streams = [
        Accum(
            step_graph,
            Reshape(
                step_graph,
                (unchunked_expert_feature_streams, i),
                tile_N,
                0,
                write_back_mu=False,
                pad_fn=init_fn.Zero(shape=(1, D), dtype=Float32()),
            ),
            Tile(tile_dtype=Float32(), shape=(tile_N, D)),
            accum_fn.RetileRow(),
            init_fn.Empty(shape=(0, D), dtype=Float32()),
            1,
            False,
            1024,
        )
        for i in range(model_config.n_routed_experts)
    ]  # [(Dyn + tile_N -1) // tile_N] of tile_N x D

    # ------------ Stage 5: Repeat input features ------------
    # - input stream shape:   [(Dyn + tile_N -1) // tile_N]              x n_routed_experts (tile: [tile_N, D])
    # - output stream shape:  [(Dyn + tile_N -1) // tile_N, F // tile_F] x n_routed_experts (tile: [tile_N, D])
    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            expert_feature_streams[i],
            repeat_factor=F // tile_F,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 5.5: Group input features (EagerMerge) ------------
    # - input stream shape:         [ (Dyn + tile_N -1) // tile_N, F // tile_F] x n_routed_experts (tile: [tile_N, D])
    # - output stream shape (data): [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_N, D])
    # - output stream shape (sel):  [∑((Dyn + tile_N -1) // tile_N)]              x (n_routed_experts // group_size) (dtype: Multihot)
    #                              (Here, the sum expresses the sum between the experts grouped together)
    grouped_feature_streams: List[EagerMerge] = []
    curr_group: List[RepeatStatic] = []
    merge_rank = repeated_feature_streams[0].stream.rank
    for i in range(model_config.n_routed_experts):
        # group every group_size experts together and create an EagerMerge op
        curr_group.append(repeated_feature_streams[i])
        if i % group_size == group_size - 1:
            grouped_feature_streams.append(
                EagerMerge(step_graph, curr_group, merge_rank)  # type: ignore (Cannot infer RepeatStatic is a subclass of StepOps)
            )
            curr_group = []

    # ------------ Stage 6: Load up parameters ------------
    # - tensor shape: [D, F]
    # - ref stream shape:    [(Dyn + tile_N -1) // tile_N]
    # - per tensor stream shape:                          [F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape: [(Dyn + tile_N -1) // tile_N, F // tile_F, 1] (tile: [D, tile_F])
    up_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=w_up_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [(Dyn + tile_N -1) // tile_N, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape:  [(Dyn + tile_N -1) // tile_N, F // tile_F]    (tile: [D, tile_F])
    ready_up_loads = [
        Flatten(
            graph=step_graph,
            input=up_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 6.5: Reassemble the up weights for the currently executed expert ------------
    # Shape information for each expert group (total number of expert groups: n_routed_experts // group_size)
    # - input stream shape:   [  (Dyn + tile_N -1) // tile_N, F // tile_F] x group_size (tile: [D, tile_F])
    # - control stream shape: [∑((Dyn + tile_N -1) // tile_N)]                         (dtype: Multihot)
    # - output stream shape:  [∑((Dyn + tile_N -1) // tile_N), DynDim, F // tile_F]    (tile: [D, tile_F]) (The DynDim is actually 1)
    reassembled_up_loads = [
        FlatReassemble(
            step_graph,
            ready_up_loads[i * group_size : (i + 1) * group_size],  # type: ignore (Cannot infer Flatten is a subclass of StepOps)
            (grouped_feature_streams[i], 1),  # idx: 0 (data), 1(sel)
            reassemble_rank=1,
            switch_cycles=[1 for _ in range(group_size)],
            write_back_mu=False,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    # As we have the information that all the MultiHot in the control stream is a one-hot,
    # we can replace the newly inserted DynDim with 1.
    # (FlatReassemble inserts a new DynDim by default as it does not have the information of the data in the control stream)
    # (Keeping this flexibility is intentional as this allows also expressing models that choose different number of experts at a time
    # or control flow that chooses different number of branches at a time)
    for reassembled_up_load_i in reassembled_up_loads:
        reassembled_up_load_i.stream.shape = (
            reassembled_up_load_i.stream.shape[:-2]
            + (1,)
            + reassembled_up_load_i.stream.shape[-1:]
        )

    # - input stream shape:  [∑((Dyn + tile_N -1) // tile_N), 1, F // tile_F] x n_routed_experts // group_size (tile: [D, tile_F])
    # - output stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F]    x n_routed_experts // group_size (tile: [D, tile_F])
    flattened_reassembled_up_loads = [
        Flatten(
            graph=step_graph,
            input=reassembled_up_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    # ------------ Stage 7: Compute the up features ------------
    # - input stream shape:  [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_N, D])
    # - weight stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [D, tile_F])
    # - output stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_N, tile_F])
    up_feature_streams = [
        BinaryMap(
            step_graph,
            (feature, 0),  # idx: 0 (data), 1(sel)
            weight,
            map_fn.Matmul(weight_transposed=False),
            False,
            up_compute_bw,
        )
        for feature, weight in zip(
            grouped_feature_streams, flattened_reassembled_up_loads
        )
    ]

    # ------------ Stage 8: Load gate parameters ------------
    # - tensor shape: [D, F]
    # - ref stream shape:    [(Dyn + tile_N -1) // tile_N]
    # - per tensor stream shape:                          [F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape: [(Dyn + tile_N -1) // tile_N, F // tile_F, 1] (tile: [D, tile_F])
    gate_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=w_gate_list[i],
            stride=(1, D // D),
            out_shape_tiled=(F // tile_F, 1),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:  [(Dyn + tile_N -1) // tile_N, F // tile_F, 1] (tile: [D, tile_F])
    # - output stream shape: [(Dyn + tile_N -1) // tile_N, F // tile_F]    (tile: [D, tile_F])
    ready_gate_loads = [
        Flatten(
            graph=step_graph,
            input=gate_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 7.5: Reassemble the gate weights for the currently executed expert ------------
    # Shape information for each expert group
    # - input stream shape:   [  (Dyn + tile_N -1) // tile_N, F // tile_F] x group_size (tile: [D, tile_F])
    # - control stream shape: [∑((Dyn + tile_N -1) // tile_N)]                         (dtype: Multihot)
    # - output stream shape:  [∑((Dyn + tile_N -1) // tile_N), DynDim, F // tile_F]    (tile: [D, tile_F]) (The DynDim is actually 1)
    reassembled_gate_loads = [
        FlatReassemble(
            step_graph,
            ready_gate_loads[i * group_size : (i + 1) * group_size],  # type: ignore (Cannot infer Flatten is a subclass of StepOps)
            (grouped_feature_streams[i], 1),  # idx: 0 (data), 1(sel)
            reassemble_rank=1,
            switch_cycles=[1 for _ in range(group_size)],
            write_back_mu=False,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    for reassembled_gate_load_i in reassembled_gate_loads:
        reassembled_gate_load_i.stream.shape = (
            reassembled_gate_load_i.stream.shape[:-2]
            + (1,)
            + reassembled_gate_load_i.stream.shape[-1:]
        )

    # - input stream shape:  [∑((Dyn + tile_N -1) // tile_N), 1, F // tile_F] x n_routed_experts // group_size (tile: [D, tile_F])
    # - output stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F]    x n_routed_experts // group_size (tile: [D, tile_F])
    flattened_reassembled_gate_loads = [
        Flatten(
            graph=step_graph,
            input=reassembled_gate_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    # ------------ Stage 8: Compute the gate features ------------
    # - input stream shape:  [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_N, D])
    # - weight stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [D, tile_F])
    # - output stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_N, tile_F])
    pre_act_gate_feature_streams = [
        BinaryMap(
            step_graph,
            (feature, 0),  # idx: 0 (data), 1(sel)
            weight,
            map_fn.Matmul(weight_transposed=False),
            False,
            gate_compute_bw,
        )
        for feature, weight in zip(
            grouped_feature_streams, flattened_reassembled_gate_loads
        )
    ]

    # ------------ Stage 9: Compute the activation ------------
    # - input stream shape:   [∑((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    # - output stream shape:  [∑((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
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
    # - input1 stream shape:   [∑((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N1, tile_F])
    # - input2 stream shape:   [∑((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    # - output stream shape:   [∑((Dyn + tile_N -1) // tile_N), F // tile_F] (tile: [tile_N, tile_F])
    projected_feature_streams = [
        BinaryMap(
            step_graph, up_feature, gate_feature, map_fn.Mul(), False, mult_compute_bw
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]

    # ------------ Stage 11: Load down parameters ------------
    # - tensor shape: [F, D]
    # - ref stream shape:    [(Dyn + tile_N -1) // tile_N]
    # - per tensor stream shape:                          [F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape: [(Dyn + tile_N -1) // tile_N, F // tile_F, 1] (tile: [tile_F, D])
    down_loads = [
        DynOffChipLoad(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=w_down_list[i],
            stride=(D // D, 1),
            out_shape_tiled=(F // tile_F, D // D),
            tile_row=tile_F,
            tile_col=D,
            par_dispatch=4,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # - input stream shape:   [(Dyn + tile_N -1) // tile_N, F // tile_F, 1] (tile: [tile_F, D])
    # - output stream shape:  [(Dyn + tile_N -1) // tile_N, F // tile_F]    (tile: [tile_F, D])
    ready_down_loads = [
        Flatten(
            graph=step_graph,
            input=down_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts)
    ]

    # ------------ Stage 11.5: Reassemble the down weights for the currently executed expert ------------
    # Shape information for each expert group
    # - input stream shape:   [ (Dyn + tile_N -1) // tile_N, F // tile_F] x group_size (tile: [tile_F, D])
    # - control stream shape: [∑((Dyn + tile_N -1) // tile_N)]                      (dtype: Multihot)
    # - output stream shape:  [∑((Dyn + tile_N -1) // tile_N), DynDim, F // tile_F] (tile: [tile_F, D]) (The DynDim is actually 1)
    reassembled_down_loads = [
        FlatReassemble(
            step_graph,
            ready_down_loads[i * group_size : (i + 1) * group_size],  # type: ignore (Cannot infer Flatten is a subclass of StepOps)
            (grouped_feature_streams[i], 1),  # idx: 0 (data), 1(sel)
            reassemble_rank=1,
            switch_cycles=[1 for _ in range(group_size)],
            write_back_mu=False,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    for reassembled_down_load_i in reassembled_down_loads:
        reassembled_down_load_i.stream.shape = (
            reassembled_down_load_i.stream.shape[:-2]
            + (1,)
            + reassembled_down_load_i.stream.shape[-1:]
        )

    # - input stream shape:  [∑((Dyn + tile_N -1) // tile_N), 1, F // tile_F] x n_routed_experts // group_size (tile: [tile_F, D])
    # - output stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F]    x n_routed_experts // group_size (tile: [tile_F, D])
    flattened_reassembled_down_loads = [
        Flatten(
            graph=step_graph,
            input=reassembled_down_loads[i],
            min_rank=0,
            max_rank=1,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    # ------------ Stage 12: Compute the down features ------------
    # - input stream shape:  [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_N, tile_F])
    # - weight stream shape: [∑((Dyn + tile_N -1) // tile_N), F // tile_F] x (n_routed_experts // group_size) (tile: [tile_F, D])
    # - output stream shape: [∑((Dyn + tile_N -1) // tile_N)             ] x (n_routed_experts // group_size) (tile: [tile_N, D])
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
        for feature, weight in zip(
            projected_feature_streams, flattened_reassembled_down_loads
        )
    ]

    # ------------ Stage 12.5: Partition & Retile outputs for each expert ------------
    # - input stream shape:   [∑((Dyn + tile_N -1) // tile_N)] x (n_routed_experts // group_size) (tile: [tile_N, D])
    # - control stream shape: [∑((Dyn + tile_N -1) // tile_N)] x (n_routed_experts // group_size) (dtype: Multihot)
    # - output stream shape:  [  (Dyn + tile_N -1) // tile_N ] x  n_routed_experts                (tile: [tile_N, D])
    output_per_expert = [
        FlatPartition(
            step_graph,
            chunked_down_feature_streams[i],
            (grouped_feature_streams[i], 1),  # idx: 0 (data), 1(sel)
            partition_rank=0,
            switch_cycles=[1 for _ in range(group_size)],
            write_back_mu=False,
            num_consumers=group_size,
        )
        for i in range(model_config.n_routed_experts // group_size)
    ]

    # replace the dynamic dims with the dyndims used before the eager merge
    for expert_group_idx, op in enumerate(output_per_expert):
        op: FlatPartition
        for expert_idx, expert_stream in enumerate(op.stream_list):
            expert_stream.shape = (
                expert_feature_streams[
                    expert_group_idx * group_size + expert_idx
                ].stream.shape[0],
            ) + expert_stream.shape[1:]

    # - input stream shape:  [(Dyn + tile_N -1) // tile_N] x  n_routed_experts (tile: [tile_N, D])
    # - output stream shape: [Dyn_retile]                  x  n_routed_experts (tile: [1, D])
    #   (after replacement): [Dyn]                         x  n_routed_experts (tile: [1, D])
    down_feature_streams = [
        RetileStreamify(
            graph=step_graph,
            input=(output_per_expert[i // group_size], i % group_size),
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
        Promote(
            graph=step_graph,
            input=accumed_stream,
            promote_rank=0,
        ),
        par_dispatch=4,
        store_file_name="output",
    )  # [1, N, 1] (tile: [1, D])

    return output


def call_hybrid_moe_gemm(
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
    tile_N: int,
    tile_F: int,
    group_size: int,
    save_graph: bool,
    simulate_rust: str,
    logging: Optional[str] = None,
) -> tuple[StepOps, sympy.Expr, sympy.Expr, int, int]:
    step_graph = MultiDiGraph()

    output: OffChipStore = hybrid_moe_gemm(
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
        group_size=group_size,
    )

    print(f"Output untiled: {output.get_untiled_shape()}")

    step_graph = infer_broadcast(step_graph)

    if save_graph:
        OUTPUT_FILENAME = "hybrid_moe_gemm"
        save_graph_format(step_graph, OUTPUT_FILENAME, ["svg"])

    total_off_chip_traffic = sympy.Integer(0)
    total_on_chip_requirement = sympy.Integer(0)

    for node_tuple in step_graph.nodes(data=True):
        node, data = node_tuple
        if isinstance(node, StepOps):
            total_off_chip_traffic = sympy.Add(
                total_off_chip_traffic, node.off_chip_traffic()
            )
            total_on_chip_requirement = sympy.Add(
                total_on_chip_requirement, node.on_chip_requirement()
            )
        else:
            raise ValueError(f"Node {node} in the graph is not a StepOps")

    cycles = 0
    duration_ms = 0
    duration_s = 0

    if simulate_rust in ["full", "timing"]:
        hbm_config = HBMConfig(64, 8, 2, 2, 1, 14)
        sim_config = SimConfig(channel_depth=1, functional_sim=simulate_rust == "full")

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
        sympy.simplify(total_off_chip_traffic),
        sympy.simplify(total_on_chip_requirement),
        cycles,
        duration_s,
    )


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
    dim = 64  # 2048 // 32 = 64
    moe_inter_dim = 352  # 1408 // 4 (Can use tile size of 32)


@dataclass
class SmallerMixtral:  # 64x scaled down version for each dimension
    n_routed_experts = 8
    n_activated_experts = 2
    dim = 64  # 4096/64
    moe_inter_dim = 224  # 14336/64 (Can use tile size upto 32)


@dataclass
class Mixtral8x7b:
    n_routed_experts = 8
    n_activated_experts = 2
    dim = 4096
    moe_inter_dim = 14336


@dataclass
class Qwen30b:
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 2048  # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json#L12
    moe_inter_dim = (
        768  # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json#L19
    )


@dataclass
class SmallerQwen30b:
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 256
    moe_inter_dim = 48


def get_expert_selection(
    B: int, model_config, seed: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is None:
        # Calculate ideal total that's divisible by number of experts
        ideal_per_expert = (
            B * model_config.n_activated_experts
        ) // model_config.n_routed_experts
        total_selections = ideal_per_expert * model_config.n_routed_experts

        # Generate perfectly balanced expert assignments
        expert_list = []
        for expert_id in range(model_config.n_routed_experts):
            expert_list.extend([expert_id] * ideal_per_expert)

        # Shuffle and truncate/pad to fit your batch structure
        import random

        random.shuffle(expert_list)

        # Reshape to fit your original structure (may need padding/truncating)
        if len(expert_list) != B * model_config.n_activated_experts:
            # Handle the mismatch - either pad or truncate
            target_length = B * model_config.n_activated_experts
            if len(expert_list) < target_length:
                # Pad by cycling through experts
                while len(expert_list) < target_length:
                    expert_list.append(
                        expert_list[len(expert_list) % model_config.n_routed_experts]
                    )
            else:
                # Truncate
                expert_list = expert_list[:target_length]

        expert_indices = torch.tensor(expert_list).reshape(
            B, model_config.n_activated_experts
        )

        expert_counts = torch.bincount(
            expert_indices.flatten(), minlength=model_config.n_routed_experts
        )

        return (expert_indices, expert_counts)
    else:
        torch.manual_seed(seed)
        # [B, n_activated_experts]
        expert_indices = torch.topk(
            torch.randn(B, model_config.n_routed_experts),
            model_config.n_activated_experts,
            dim=-1,
        )[1]

        # Get bincount across all batches [n_routed_experts]
        expert_counts = torch.bincount(
            expert_indices.flatten(), minlength=model_config.n_routed_experts
        )

        return (expert_indices, expert_counts)


def run_hybrid_moe_gemm(
    tile_N: int,
    tile_F: int,
    group_size: int,
    input_tensor,
    expert_indices,
    model_config,
    simulate_rust,  # either "full", "timing", None
    gold_check,
):

    B = expert_indices.shape[0]

    # ------------ Group Size ------------
    GROUP_SIZE = group_size

    # ------------ Tile Configuration ------------
    tile_N = tile_N  # M (The number of requests to chunk for the GEMM in each expert)
    tile_F = tile_F  # K (The tile size used for the model_config.dim dimension)
    # For the model_config.moe_inter_dim dimension, we don't tile
    # For Gate & Up linear, we use TileMN. For Down linear, we use TileMK.

    # ------------ Compute Bandwidths ------------
    GATE_COMPUTE_BW = 1022
    UP_COMPUTE_BW = 1022
    ACT_FN_COMPUTE_BW = 1022
    MULT_COMPUTE_BW = 1022
    DOWN_COMPUTE_BW = 1022
    WEIGHT_SCALE_COMPUTE_BW = 1022
    ACCUM_COMPUTE_BW = 1022

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

    output, off_chip_traffic, on_chip_requirement, cycles, duration_s = call_hybrid_moe_gemm(  # type: ignore (Cannot infer type of output properly)
        model_config=model_config,
        batch=B,
        gate_compute_bw=GATE_COMPUTE_BW * GROUP_SIZE,
        up_compute_bw=UP_COMPUTE_BW * GROUP_SIZE,
        act_fn_compute_bw=ACT_FN_COMPUTE_BW * GROUP_SIZE,
        mult_compute_bw=MULT_COMPUTE_BW * GROUP_SIZE,
        down_compute_bw=DOWN_COMPUTE_BW * GROUP_SIZE,
        weight_scale_compute_bw=WEIGHT_SCALE_COMPUTE_BW,
        accum_compute_bw=ACCUM_COMPUTE_BW,
        input_tensor=input_tensor,
        expert_multihot=expert_multihot,
        expert_onehot=expert_onehot,
        expert_weights=expert_weights,
        w_gate_list=w_gate_list,
        w_up_list=w_up_list,
        w_down_list=w_down_list,
        tile_F=tile_F,
        tile_N=tile_N,
        group_size=GROUP_SIZE,
        save_graph=False,
        simulate_rust=simulate_rust,
        # logging="expert_par_gemm_reshape",
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


def test_group_size_sweep():
    # ------------ Model Configuration ------------
    # model_config = Qwen30b()
    # model_config = SmallerMixtral()
    # model_config = SmallerDeepSeekV3()
    model_config = SmallerQwen30b()
    tile_N = 16  # Size of grouped requests for each expert
    tile_F = 16  # MoE intermediate dimension

    # Same iteration, Different layer
    iter_list = [22]  # 1-500, 2000-2100, 3000-3100, 4000-4100
    layer_list = [2]  # [0, 1, 2, 10, 20, 30, 40, 46, 47]

    # Same layer, Different iteration
    # iter = [0,4,8,12,16,20, 24]
    # layer = 20

    group_size_list = [1]  # [1, 2, 4, 8, 16, 32]

    for iter in iter_list:
        for layer in layer_list:
            results = []
            for group_size in group_size_list:
                expert_selection_file = f"/home/ginasohn/expert_routing/processed_qwen/continuous_batching_80gb_max4192_per_layer/iter_{iter:03d}_layer_{layer:03d}.npz"
                expert_indices_npz = np.load(expert_selection_file)
                expert_indices = torch.from_numpy(
                    expert_indices_npz["data"]
                )  # [B, n_activated_experts]

                B = expert_indices.shape[0]

                # ------------ Input generation ------------
                # Set the random seed
                seed = 5
                torch.manual_seed(seed)

                input_tensor = torch.randn(B, model_config.dim)

                # ------------ Expert Indices ------------
                # expert_indices: [B, n_activated_experts]
                # expert_counts: [n_routed_experts] (bincount across all batches)
                expert_counts = torch.bincount(
                    expert_indices.flatten(), minlength=model_config.n_routed_experts
                )
                print(f"Expert counts: {expert_counts}")

                off_chip_traffic, on_chip_requirement, cycles, duration_s = (
                    run_hybrid_moe_gemm(
                        tile_N=tile_N,
                        tile_F=tile_F,
                        group_size=group_size,
                        input_tensor=input_tensor,
                        expert_indices=expert_indices,
                        model_config=model_config,
                        simulate_rust="timing",
                        gold_check=False,
                    )
                )

                # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
                num_tiles = [
                    (routed_toks + tile_N - 1) // tile_N
                    for routed_toks in expert_counts.tolist()
                ]
                after_pad_batch_dim = [
                    num_tiles_i * tile_N for num_tiles_i in num_tiles
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

                free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

                sub_dict = {
                    symbol: value
                    for symbol, value in zip(free_symbols, expert_counts.tolist())
                }

                off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

                results.append(
                    {
                        "group_size": group_size,
                        "cycles": cycles,
                        "flops": flops,
                        "duration_s": duration_s,
                        "off_chip_traffic_bytes": off_chip_traffic_val,
                        "on_chip_requirement": on_chip_requirement,
                    }
                )

            out_file = f"qwen_256_48_80gb_max4192_iter{iter:03d}_layer_{layer:03d}_n{tile_N}_f{tile_F}.csv"

            try:
                with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "group_size",
                        "cycles",
                        "flops",
                        "duration_s",
                        "off_chip_traffic_bytes",
                        "on_chip_requirement",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()

                    # Write data rows
                    for result in results:
                        writer.writerow(result)

                print(f"Results written to {out_file}")
            except Exception as e:
                print(f"Error writing CSV file: {e}")
