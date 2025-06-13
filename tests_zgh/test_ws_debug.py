import torch
from networkx import MultiDiGraph
from step_py.functions import map_fn, init_fn
from step_py.ops import *
from step_py.utility_ops import *
from rewrite.broadcast import infer_broadcast
from utils.draw_graph import save_graph_format
from sim import simulate, HBMConfig

def step_impl(
    expert_ups,
    expert_downs,
    input_tensor,
    indices,
    n_activated,
    E,
    N,
    D,
    F,
    tile_F,
):
    step_graph: MultiDiGraph = MultiDiGraph()
    offchip_par_dispatch = 4
    COMPUTE_BW = 8
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
    feature_select_gen = SelectGen(
        is_multihot=True,
        tensor=expert_selection_multi_hot,
    )  # [1, N]

    # Stage 3: Load the weights

    # Stage 4: Partition the input feature stream
    expert_feature_streams = FlatPartition(
        step_graph,
        input_load,  # [1, N]
        feature_select_gen,  # [1, N]
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
            COMPUTE_BW,
        )
        for feature, weight in zip(repeated_feature_streams, ready_up_loads)
    ]  # [Dyn, F // tile_F]

    # Stage 8: Repeat gate parameters

    # Stage 9: Compute the gate features

    # Stage 10: Compute the projected features

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
    projected_feature_streams = up_feature_streams
    down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
            init_fn.Zero(shape=(1, D), dtype=Float32()),
            1,
            False,
            COMPUTE_BW,
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ]  # [Dyn]

    # Stage 13: Partition the scalar weights

    # Post Stage 13: Align the dynamic shapes with down_feature_streams

    # Stage 14: Compute the weighted features

    # Stage 15: Reassemble the weighted features
    weighted_feature_streams = down_feature_streams
    reassembled_stream = FlatReassemble(
        step_graph,
        weighted_feature_streams,  # [Dyn]
        feature_select_gen,  # [1, N]
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
    )  # [1, N, Ragged]

    # Stage 16: Accumulate the reassembled features

    # Stage 17: Store the output
    accumed_stream = reassembled_stream
    accumed_stream._stream.shape =  accumed_stream._stream.shape[:-1] + (n_activated,)
    print(f"Final shape: {accumed_stream._stream.shape}")
    output = OffChipStore(
        step_graph,
        accumed_stream,
        par_dispatch=offchip_par_dispatch,
        store_file_name="output",
    )  # [1, N, Ragged]
    
    step_graph = infer_broadcast(step_graph)
    OUTPUT_FILENAME = "moe_weight_stationary_debug"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])
    
    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/zgh23/step_tl/graph.pb",
    )
    
    return output

def test_expert():
    # N, D, F, E = 3, 8, 6, 5
    # tile_F = 3
    N, D, F, E = 19, 64, 128, 8
    tile_F = 16
    n_activated = 2
    weights = torch.randn(N, E).softmax(dim=-1)
    _, indices = torch.topk(weights, n_activated, dim=-1)
    input_tensor = torch.randn(N, D)
    expert_ups = [torch.randn(D, F) for _ in range(E)]
    expert_downs = [torch.randn(F, D) for _ in range(E)]
    output = step_impl(
        expert_ups,
        expert_downs,
        input_tensor,
        indices,
        n_activated,
        E,
        N,
        D,
        F,
        tile_F,
    )
    print(f"Output shape: {output.get_untiled_shape()}")

test_expert()