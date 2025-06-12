import torch
from networkx import MultiDiGraph
from step_py.functions import map_fn
from step_py.ops import *
from step_py.utility_ops import *
from rewrite.broadcast import infer_broadcast
from utils.draw_graph import save_graph_format


def pytorch_ref(
    expert_ups, expert_downs, expert_gates, input_tensor, indices, weights, E, N, D
) -> OffChipStore:
    output_tensor = torch.zeros(N, D)
    counts = torch.bincount(indices.flatten(), minlength=E).tolist()
    for i in range(E):
        if counts[i] == 0:
            continue
        idx, top = torch.where(indices == i)
        output_tensor[idx] += weights[idx, top, None] * (
            (
                (input_tensor[idx] @ expert_ups[i])
                * torch.nn.functional.silu(input_tensor[idx] @ expert_gates[i])
            )
            @ expert_downs[i]
        )
    return output_tensor


def step_impl(
    expert_ups,
    expert_downs,
    expert_gates,
    input_tensor,
    indices,
    weights,
    E,
    N,
    D,
    F,
    tile_F,
):
    step_graph: MultiDiGraph = MultiDiGraph()
    offchip_par_dispatch = 4

    # Stage 1: Load the parameters and input tensors
    up_loads = [
        OffChipLoad(
            underlying=expert_ups[i],
            stride=(1,),
            out_shape_tiled=(F // tile_F,),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=offchip_par_dispatch,
        )
        for i in range(E)
    ]  # [1, F // tile_F]

    down_loads = [
        OffChipLoad(
            underlying=expert_downs[i],
            stride=(1,),
            out_shape_tiled=(F // tile_F,),
            tile_row=tile_F,
            tile_col=D,
            par_dispatch=offchip_par_dispatch,
        )
        for i in range(E)
    ]  # [1, F // tile_F]

    gate_loads = [
        OffChipLoad(
            underlying=expert_gates[i],
            stride=(1,),
            out_shape_tiled=(F // tile_F,),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=offchip_par_dispatch,
        )
        for i in range(E)
    ]  # [1, F // tile_F]

    input_load = OffChipLoad(
        underlying=input_tensor,
        stride=(1,),
        out_shape_tiled=(N,),
        tile_row=1,
        tile_col=D,
        par_dispatch=offchip_par_dispatch,
    )  # [1, N]

    # Stage 2: Generate the selection stream
    expert_selection_one_hot = torch.nn.functional.one_hot(indices, E)
    expert_selection_multi_hot = expert_selection_one_hot.sum(dim=-2)
    select_gen = SelectGen(
        is_multihot=True,
        tensor=expert_selection_multi_hot,
    )  # [1, N]

    # Stage 3: Load the weights
    weights_load = OffChipLoad(
        underlying=weights,
        stride=(1,),
        out_shape_tiled=(N,),
        tile_row=1,
        tile_col=E,
        par_dispatch=offchip_par_dispatch,
    )  # [1, N]

    # Stage 4: Partition the input feature stream
    expert_feature_streams = FlatPartition(
        step_graph,
        input_load,  # [1, N]
        select_gen,  # [1, N]
        partition_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
        num_consumers=E,
    )  # [Dyn]

    # Stage 5: Repeat input features
    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            (expert_feature_streams, i),
            repeat_factor=F // tile_F,
        )
        for i in range(E)
    ]  # [Dyn, F // tile_F]

    # Stage 6: Repeat up parameters
    deranked_up_loads = [
        Bufferize(step_graph, stream, 1) for stream in up_loads
    ]  # [1,]
    ready_up_loads = [
        DynStreamify(
            step_graph,
            deranked_up_loads[i],  # [1,] of [F // tile_F]
            (expert_feature_streams, i),  # [Dyn,]
            0,
            1,
        )
        for i in range(E)
    ]  # [Dyn, F // tile_F]

    # Stage 7: Compute the up features
    up_feature_streams = [
        BinaryMap(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )
        for feature, weight in zip(repeated_feature_streams, ready_up_loads)
    ]  # [Dyn, F // tile_F]

    # Stage 8: Repeat gate parameters
    deranked_gate_loads = [
        Bufferize(step_graph, stream, 1) for stream in gate_loads
    ]  # [1,]
    ready_gate_loads = [
        DynStreamify(
            step_graph,
            deranked_gate_loads[i],  # [1,]
            (expert_feature_streams, i),  # [Dyn,]
            0,
            1,
        )
        for i in range(E)
    ]  # [Dyn, F // tile_F]

    # Stage 9: Compute the gate features
    gate_feature_streams = [
        UnaryMap(
            step_graph,
            BinaryMap(
                step_graph,
                feature,
                weight,
                map_fn.Matmul(weight_transposed=False),
                False,
                1024,
            ),
            map_fn.Silu(),
            False,
            1024,
        )
        for feature, weight in zip(repeated_feature_streams, ready_gate_loads)
    ]  # [Dyn, F // tile_F]

    # Stage 10: Compute the projected features
    projected_feature_streams = [
        BinaryMap(step_graph, up_feature, gate_feature, map_fn.Mul(), False, 1024)
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ]  # [Dyn, F // tile_F]

    # Stage 11: Repeat down parameters
    deranked_down_loads = [
        Bufferize(step_graph, stream, 1) for stream in down_loads
    ]  # [1,]
    ready_down_loads = [
        DynStreamify(
            step_graph,
            deranked_down_loads[i],  # [1,]
            (expert_feature_streams, i),  # [Dyn,]
            0,
            1,
        )
        for i in range(E)
    ]  # [Dyn, F // tile_F]

    # Stage 12: Compute the down features
    down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            1,
            False,
            1024,
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ]  # [Dyn]

    # Stage 13: Partition the scalar weights
    expert_weight_streams = FlatPartition(
        step_graph,
        weights_load,
        select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
        num_consumers=E,
    )  # [Dyn]

    # Stage 14: Compute the weighted features
    weighted_feature_streams = [
        BinaryMap(
            step_graph,
            down_feature_streams[i],
            (expert_weight_streams, i),
            map_fn.SelectMul(0, i),
            False,
            1024,
        )
        for i in range(E)
    ]  # [Dyn]

    # Stage 15: Reassemble the weighted features
    reassembled_stream = FlatReassemble(
        step_graph,
        weighted_feature_streams,  # [Dyn]
        select_gen,  # [1, N]
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
    )  # [1, N, Ragged]

    # Stage 16: Accumulate the reassembled features
    accumed_stream = Accum(
        step_graph,
        reassembled_stream,  # [1, N, Ragged]
        Tile(tile_dtype=Float32(), shape=(1, D)),
        map_fn.Add(),
        1,
        False,
        1024,
    )  # [1, N]

    # Stage 17: Store the output
    output = OffChipStore(
        step_graph,
        accumed_stream,
        par_dispatch=offchip_par_dispatch,
        store_file_name="output",
    )  # [1, N]

    step_graph = infer_broadcast(step_graph)
    OUTPUT_FILENAME = "moe_weight_stationary"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])
    return output


def test_expert():
    # N, D, F, E = 3, 8, 6, 5
    N, D, F, E = 19, 64, 128, 8
    tile_F = 16
    n_activated = 2
    weights = torch.randn(N, E).softmax(dim=-1)
    selected_weights, indices = torch.topk(weights, n_activated, dim=-1)
    input_tensor = torch.randn(N, D)
    expert_ups = [torch.randn(D, F) for _ in range(E)]
    expert_downs = [torch.randn(F, D) for _ in range(E)]
    expert_gates = [torch.randn(D, F) for _ in range(E)]
    gold = pytorch_ref(
        expert_ups,
        expert_downs,
        expert_gates,
        input_tensor,
        indices,
        selected_weights,
        E,
        N,
        D,
    )
    output = step_impl(
        expert_ups,
        expert_downs,
        expert_gates,
        input_tensor,
        indices,
        selected_weights,
        E,
        N,
        D,
        F,
        tile_F,
    )
    print(f"Gold shape: {gold.shape}")
    print(f"Output shape: {output.get_untiled_shape()}")


test_expert()
