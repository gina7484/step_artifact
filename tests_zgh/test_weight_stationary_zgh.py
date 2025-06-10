import torch
from networkx import MultiDiGraph
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
    ]

    down_loads = [
        OffChipLoad(
            underlying=expert_downs[i],
            stride=(1, D // tile_D),
            out_shape_tiled=(D // tile_D, 1),
            tile_row=tile_F,
            tile_col=D,
            par_dispatch=offchip_par_dispatch,
        )
        for i in range(E)
    ]

    gate_loads = [
        OffChipLoad(
            underlying=expert_gates[i],
            stride=(F // tile_F, 1),
            out_shape_tiled=(1, F // tile_F),
            tile_row=D,
            tile_col=tile_F,
            par_dispatch=offchip_par_dispatch,
        )
        for i in range(E)
    ]

    input_load = OffChipLoad(
        underlying=input_tensor,
        stride=(1, N),
        out_shape_tiled=(N, 1),
        tile_row=1,
        tile_col=D,
        par_dispatch=offchip_par_dispatch,
    )

    expert_selection_one_hot = torch.nn.functional.one_hot(indices, E)
    expert_selection_multi_hot = expert_selection_one_hot.sum(dim=-2)
    select_gen_feature = SelectGen(
        is_multihot=True,
        tensor=expert_selection_multi_hot,
    )
    select_gen_weights = SelectGen(
        is_multihot=True,
        tensor=weights,
    )

    weights_load = OffChipLoad(
        underlying=weights,
        stride=(1, 1),
        out_shape_tiled=(N, 1),
        tile_row=1,
        tile_col=E,
        par_dispatch=offchip_par_dispatch,
    )
    
    expert_feature_streams = FlatPartition(
        step_graph,
        input_load,
        select_gen_feature,
        partition_rank=1,
        switch_cycles=[1 for _ in range(E)],
        write_back_mu=False,
        num_consumers=E
    )

    repeated_feature_streams = [
        RepeatStatic(
            step_graph,
            stream,
            repeat_count=F // tile_F,
        )
        for stream in expert_feature_streams
    ]
    



    



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