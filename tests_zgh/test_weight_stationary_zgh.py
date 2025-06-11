import torch
from networkx import MultiDiGraph
from step_py.functions import map_fn
from step_py.ops import *
from step_py.utility_ops import *

def pytorch_ref(expert_ups, expert_downs, expert_gates, input_tensor, indices, weights, E, N, D):
    output_tensor = torch.zeros(N, D)
    counts = torch.bincount(indices.flatten(), minlength=E).tolist()
    for i in range(E):
        if counts[i] == 0:
            continue
        idx, top = torch.where(indices == i)
        output_tensor[idx] += weights[idx, top, None] * (
            ((input_tensor[idx] @ expert_ups[i]) * torch.nn.functional.silu(input_tensor[idx] @ expert_gates[i])) @ expert_downs[i]
        )
    return output_tensor

def step_impl(expert_ups, expert_downs, expert_gates, input_tensor, indices, weights, E, N, D, F, tile_D, tile_F):
    step_graph: MultiDiGraph = MultiDiGraph()
    offchip_par_dispatch = 4
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
    ] # [1, F // tile_F]

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
    ] # [1, F // tile_F]

    gate_loads = [
        OffChipLoad(
            underlying=expert_gates[i],
            stride=(1),
            out_shape_tiled=(F // tile_F,),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=offchip_par_dispatch,
        )
        for i in range(E)
    ] # [1, F // tile_F]

    input_load = OffChipLoad(
        underlying=input_tensor,
        stride=(1,),
        out_shape_tiled=(N,),
        tile_row=1,
        tile_col=D,
        par_dispatch=offchip_par_dispatch,
    ) # [1, N]

    expert_selection_one_hot = torch.nn.functional.one_hot(indices, E)
    expert_selection_multi_hot = expert_selection_one_hot.sum(dim=-2)
    select_gen = SelectGen(
        is_multihot=True,
        tensor=expert_selection_multi_hot,
    ) # [1, N]

    weights_load = OffChipLoad(
        underlying=weights,
        stride=(1,),
        out_shape_tiled=(N,),
        tile_row=1,
        tile_col=E,
        par_dispatch=offchip_par_dispatch,
    ) # [1, N]

    expert_feature_streams = FlatPartition(
        step_graph,
        input_load,
        select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
        num_consumers=E
    ) # [Dyn]


    deranked_up_loads = [
        Bufferize(
            step_graph,
            stream,
            1
        )
        for stream in up_loads
    ] # [1,]

    # Data: [1,] of [F // tile_F]
    # Ref: [Dyn,]
    ready_up_loads = [
        DynStreamify(
            step_graph,
            data_stream, # [1,]
            ref_stream, # [Dyn,]
            0,
            1,
        )
        for data_stream, ref_stream in zip(deranked_up_loads, expert_feature_streams)
    ] # [Dyn, F // tile_F]

    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            stream,
            repeat_count=F // tile_F,
        )
        for stream in expert_feature_streams
    ] # [Dyn, F // tile_F]

    up_feature_streams = [
        BinaryMap(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=False),
        )
        for feature, weight in zip(repeated_feature_streams, ready_up_loads)
    ] # [Dyn, F // tile_F]

    # Do the same for gate_loads
    deranked_gate_loads = [
        Bufferize(
            step_graph,
            stream,
            1
        )
        for stream in gate_loads
    ] # [1,]
    ready_gate_loads = [
        DynStreamify(
            step_graph,
            data_stream, # [1,]
            ref_stream, # [Dyn,]
            0,
            1,
        )
        for data_stream, ref_stream in zip(deranked_gate_loads, promoted_feature_streams)
    ] # [Dyn, F // tile_F]
    gate_feature_streams = [
        UnaryMap(
            step_graph,
            BinaryMap(
                step_graph,
                feature,
                weight,
                map_fn.Matmul(weight_transposed=False),
            ),
            map_fn.Silu()
        )
        for feature, weight in zip(repeated_feature_streams, ready_gate_loads)
    ] # [Dyn, F // tile_F]

    projected_feature_streams = [
        BinaryMap(
            step_graph,
            up_feature,
            gate_feature,
            map_fn.Mul()
        )
        for up_feature, gate_feature in zip(up_feature_streams, gate_feature_streams)
    ] # [Dyn, F // tile_F]

    deranked_down_loads = [
        Bufferize(
            step_graph,
            stream,
            1
        )
        for stream in down_loads
    ] # [1,]
    ready_down_loads = [
        DynStreamify(
            step_graph,
            data_stream, # [1,]
            ref_stream, # [Dyn,]
            0,
            1,
        )
        for data_stream, ref_stream in zip(deranked_down_loads, promoted_feature_streams)
    ] # [Dyn, F // tile_F]
    down_feature_streams = [
        BinaryMapAccum(
            step_graph,
            feature,
            weight,
            map_fn.Matmul(weight_transposed=True),
            1,
            False,
            1024
        )
        for feature, weight in zip(projected_feature_streams, ready_down_loads)
    ] # [Dyn]

    expert_weight_streams = FlatPartition(
        step_graph,
        weights_load,
        select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
        num_consumers=E
    ) # [Dyn]


    weighted_feature_streams = [
        BinaryMap(
            step_graph,
            down_feature_streams[i],
            expert_weight_streams[i],
            map_fn.SelectMul(0, i)
        )
        for i in range(E)
    ] # [Dyn]

    reassembled_stream = FlatReassemble(
        step_graph,
        weighted_feature_streams,
        in_stream_rank=0,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
    ) # [Dyn, Ragged]

    accumed_stream = Accum(
        step_graph,
        reassembled_stream,
        map_fn.Add(),
        1,
        False,
        1024
    ) # [Dyn]

    output_stream = OffChipStore(
        step_graph,
        accumed_stream,
        underlying=input_tensor,
        stride=(1,),
        out_shape_tiled=(N,),
        tile_row=1,
        tile_col=D,
        par_dispatch=offchip_par_dispatch,
    ) # [1, N]







    
    



def test_expert():
    # N, D, F, E = 3, 8, 6, 5
    N, D, F, E = 19, 64, 128, 8
    n_activated = 2
    weights = torch.randn(N, E).softmax(dim=-1)
    selected_weights, indices = torch.topk(weights, n_activated, dim=-1)
    input_tensor = torch.randn(N, D)
    expert_ups = [torch.randn(D, F) for _ in range(E)]
    expert_downs = [torch.randn(F, D) for _ in range(E)]
    expert_gates = [torch.randn(D, F) for _ in range(E)]
    output_tensor = pytorch_ref(expert_ups, expert_downs, expert_gates, input_tensor, indices, selected_weights, E, N, D)

test_expert()