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
from utils.moe_gold_calc import moe_linear_gold_calc_batched


def test_expert_selection_generation():
    B = 4
    H = 6
    E = 8

    _, expert_selection = torch.topk(torch.randn(B, H, E), 2, dim=-1)

    # pylint: disable=not-callable
    expert_selection_one_hot = torch.nn.functional.one_hot(expert_selection, E)
    expert_selection_multi_hot = expert_selection_one_hot.sum(dim=-2)

    step_graph: MultiDiGraph = MultiDiGraph()

    select_gan = SelectGen(
        is_multihot=True,
        tensor=expert_selection_multi_hot,
    )

    printer_context = PrinterContext(step_graph, select_gan)

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )


def test_prefill_expert_mnk_mnk():

    # ===================== Input Creation =====================
    B = 2
    N = 4
    EXPERT = 8  # number of experts
    H = 64
    MLP_HID = 128

    input = torch.randn(B, N, H)

    linear_gate = torch.nn.Linear(H, MLP_HID, bias=False)
    linear_up = torch.nn.Linear(H, MLP_HID, bias=False)
    linear_down = torch.nn.Linear(MLP_HID, H, bias=False)

    gold_gate = linear_gate(input)  # [B, N, MLP_HID]
    gold_up = linear_up(input)  # [B, N, MLP_HID]

    w_gate = linear_gate.weight.T.detach().clone().contiguous()  # [H, MLP_HID]
    w_up = linear_up.weight.T.detach().clone().contiguous()  # [H, MLP_HID]
    w_down = linear_down.weight.T.detach().clone().contiguous()  # [MLP_HID, H]

    _, expert_selection = torch.topk(torch.randn(B, N, EXPERT), 2, dim=-1)  # [B, N, 2]

    # pylint: disable=not-callable
    expert_selection_one_hot = torch.nn.functional.one_hot(
        expert_selection
    )  # [B, N, 2, EXPERT]

    routing_expert_selection = expert_selection_one_hot[:, :, 0, :]  # [B, N, EXPERT]

    # ===================== Expert Selection Stream =====================
    print(f"Expert selection: \n{routing_expert_selection}")
    control = SelectGen(
        is_multihot=True,
        tensor=routing_expert_selection,
    )
    print(f"Control stream shape: {control.stream.shape}")
    print(f"[1, B, N] = {[1, B, N]}")

    # ===================== Tiling Config =====================
    gate_up_tile_config = LinearTileConfig(m=1, n=32, k=32)
    down_tile_config = LinearTileConfig(m=1, n=32, k=32)

    # ===================== FLOP Allocation =====================
    GATE_COMPUTE_BW = 1022
    UP_COMPUTE_BW = 1022
    ACT_FN_COMPUTE_BW = 1022
    MULT_COMPUTE_BW = 1022
    DOWN_COMPUTE_BW = 1022

    # ===================== Create Step Graph =====================
    # ---------- Formatting Inputs ----------
    # Input
    step_graph: MultiDiGraph = MultiDiGraph()

    in_load = OffChipLoad(
        underlying=input,  # [B, N, H] => [B, N // 1, H // 32]
        stride=(
            N // gate_up_tile_config.m * H // gate_up_tile_config.k,
            H // gate_up_tile_config.k,
            1,
        ),
        out_shape_tiled=(
            B,
            N // gate_up_tile_config.m,
            H // gate_up_tile_config.k,
        ),
        tile_row=gate_up_tile_config.m,
        tile_col=gate_up_tile_config.k,
        par_dispatch=4,
    )  # [1, B, N, H // 32]

    buff_input = Bufferize(
        graph=step_graph,
        input=in_load,
        rank=1,
    )  # [1, B, N] | [H // 32]

    formatted_input = Streamify(
        graph=step_graph,
        input=buff_input,
        repeat_factor=[MLP_HID // gate_up_tile_config.n],
        rank=1,
    )  # [1, B, N, MLP_HID // 32, H // 32]

    # W_gate
    w_gate_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders_gate = []
    for i in range(EXPERT):
        expert_loaders_gate.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_gate_partition, i),  # [Di]
                underlying=w_gate,  # [H, MLP_HID] => [H // gate_up_tile_config.k, MLP_HID // gate_up_tile_config.n]
                stride=(1, MLP_HID // gate_up_tile_config.n),
                out_shape_tiled=(
                    MLP_HID // gate_up_tile_config.n,
                    H // gate_up_tile_config.k,
                ),
                tile_row=gate_up_tile_config.k,
                tile_col=gate_up_tile_config.n,
                par_dispatch=4,
            )  # [Di, MLP_HID // 32, H // 32]
        )

    REASSEMBLE_RANK = 2
    w_gate_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders_gate,
        control=control,
        reassemble_rank=REASSEMBLE_RANK,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )

    # Substitute the dynamic dim in the stream as we have a statically known number of selected experts
    w_gate_reassemble.stream.shape = (
        w_gate_reassemble.stream.shape[: -(REASSEMBLE_RANK + 1)]
        + (1,)
        + w_gate_reassemble.stream.shape[-REASSEMBLE_RANK:]
    )

    flattened_w_gate_reassemble = Flatten(
        graph=step_graph, input=w_gate_reassemble, min_rank=1, max_rank=2
    )

    w_up_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders_up = []
    for i in range(EXPERT):
        expert_loaders_up.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_up_partition, i),  # [Di]
                underlying=w_up,  # [H, MLP_HID] => [H // gate_up_tile_config.k, MLP_HID // gate_up_tile_config.n]
                stride=(1, MLP_HID // gate_up_tile_config.n),
                out_shape_tiled=(
                    MLP_HID // gate_up_tile_config.n,
                    H // gate_up_tile_config.k,
                ),
                tile_row=gate_up_tile_config.k,
                tile_col=gate_up_tile_config.n,
                par_dispatch=4,
            )  # [Di, MLP_HID // 32, H // 32]
        )

    REASSEMBLE_RANK = 2
    w_up_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders_up,
        control=control,
        reassemble_rank=REASSEMBLE_RANK,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )

    # Substitute the dynamic dim in the stream as we have a statically known number of selected experts
    w_up_reassemble.stream.shape = (
        w_up_reassemble.stream.shape[: -(REASSEMBLE_RANK + 1)]
        + (1,)
        + w_up_reassemble.stream.shape[-REASSEMBLE_RANK:]
    )

    flattened_w_up_reassemble = Flatten(
        graph=step_graph, input=w_up_reassemble, min_rank=1, max_rank=2
    )

    # # ---------- Computation ----------
    # # Linear (Gate)
    gate = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=flattened_w_gate_reassemble,
        fn=map_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(gate_up_tile_config.m, gate_up_tile_config.n),
            dtype=formatted_input.stream.stream_dtype.tile_dtype,
        ),
        rank=1,
        write_back_mu=False,
        compute_bw=GATE_COMPUTE_BW,
    )  # [1, B, N, MLP_HID // 32]

    # Activation (SiLU)
    silu_gate = UnaryMap(
        graph=step_graph,
        input=gate,
        fn=map_fn.Silu(),
        write_back_mu=False,
        compute_bw=ACT_FN_COMPUTE_BW,
    )

    # Linear (Up)
    up = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=flattened_w_up_reassemble,
        fn=map_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(gate_up_tile_config.m, gate_up_tile_config.n),
            dtype=formatted_input.stream.stream_dtype.tile_dtype,
        ),
        rank=1,
        write_back_mu=False,
        compute_bw=UP_COMPUTE_BW,
    )  # [1, B, N, MLP_HID // 32]

    # Elementwise Multiplication (Gate * Up)
    gate_up = BinaryMap(
        graph=step_graph,
        in1=silu_gate,
        in2=up,
        fn=map_fn.Mul(),
        write_back_mu=True,
        compute_bw=MULT_COMPUTE_BW,
    )  # [1, B, N, MLP_HID // 32]

    # Linear (Down)
    # bufferize 1D
    buff_gate_up = Bufferize(graph=step_graph, input=gate_up, rank=1)  # [1, B, N]
    formatted_gate_up = Streamify(
        graph=step_graph,
        input=buff_gate_up,
        repeat_factor=[H // down_tile_config.n],
        rank=1,
    )  # [1, B, N, H // 32, MLP_HID // 32]

    w_down_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders_down = []
    for i in range(EXPERT):
        expert_loaders_down.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_down_partition, i),  # [Di]
                underlying=w_down,  # [MLP_HID, H] => [MLP_HID // down_tile_config.k, H // down_tile_config.n]
                stride=(1, H // down_tile_config.n),
                out_shape_tiled=(
                    H // down_tile_config.n,
                    MLP_HID // down_tile_config.k,
                ),
                tile_row=down_tile_config.k,
                tile_col=down_tile_config.n,
                par_dispatch=4,
            )  # [Di, MLP_HID // 32, H // 32]
        )

    REASSEMBLE_RANK = 2
    w_down_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders_down,
        control=control,
        reassemble_rank=REASSEMBLE_RANK,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )

    # Substitute the dynamic dim in the stream as we have a statically known number of selected experts
    w_down_reassemble.stream.shape = (
        w_down_reassemble.stream.shape[: -(REASSEMBLE_RANK + 1)]
        + (1,)
        + w_down_reassemble.stream.shape[-REASSEMBLE_RANK:]
    )

    flattened_w_down_reassemble = Flatten(
        graph=step_graph, input=w_down_reassemble, min_rank=1, max_rank=2
    )

    # ---------- Computation ----------
    # Linear (Down)
    down = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_gate_up,
        in2=flattened_w_down_reassemble,
        fn=map_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(down_tile_config.m, down_tile_config.n),
            dtype=formatted_gate_up.stream.stream_dtype.tile_dtype,
        ),
        rank=1,
        write_back_mu=True,
        compute_bw=DOWN_COMPUTE_BW,
    )  # [1, B, N, H // 32]

    # Output
    output = OffChipStore(
        graph=step_graph,
        input=down,
        par_dispatch=4,
    )

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Check whether the shapes match ================
    # print(f"Gold (up) shape: {gold_up.shape}")
    print(f"Output shape: {output.get_untiled_shape()}")

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "input_stationary"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])


def test_incremental_prefill_expert_mnk_mnk():

    # ===================== Data Creation =====================
    B = 2
    N = 4
    EXPERT = 8  # number of experts
    H = 64
    MLP_HID = 128

    # ------------ Selection ------------
    expert_indices = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])  # [B,N]
    # pylint: disable=not-callable
    routing_expert_selection = torch.nn.functional.one_hot(
        expert_indices, num_classes=EXPERT
    )  # [B, N, EXPERT]

    # ------------ Input ------------
    input_tensor = torch.randn(B, N, H)

    # ------------ Expert Weights ------------
    linear_gate_list = [torch.nn.Linear(H, MLP_HID, bias=False) for _ in range(EXPERT)]
    linear_up_list = [torch.nn.Linear(H, MLP_HID, bias=False) for _ in range(EXPERT)]
    linear_down_list = [torch.nn.Linear(MLP_HID, H, bias=False) for _ in range(EXPERT)]

    w_gate_list = [
        linear_gate.weight.T.detach().clone().contiguous()
        for linear_gate in linear_gate_list
    ]  # [H, MLP_HID] x EXPERT
    w_up_list = [
        linear_up.weight.T.detach().clone().contiguous() for linear_up in linear_up_list
    ]  # [H, MLP_HID] x EXPERT
    w_down_list = [
        linear_down.weight.T.detach().clone().contiguous()
        for linear_down in linear_down_list
    ]  # [MLP_HID, H] x EXPERT

    # ------------ Gold calculation ------------
    linear_gate_gold = moe_linear_gold_calc_batched(
        input_tensor, expert_indices, w_gate_list
    )  # [B, N, MLP_HID]
    act_linear_gate_gold = torch.nn.functional.silu(linear_gate_gold)  # [B, N, MLP_HID]
    linear_up_gold = moe_linear_gold_calc_batched(
        input_tensor, expert_indices, w_up_list
    )  # [B, N, MLP_HID]
    gold_up = act_linear_gate_gold * linear_up_gold  # [B, N, MLP_HID]
    linear_down_gold = moe_linear_gold_calc_batched(
        gold_up, expert_indices, w_down_list
    )  # [B, N, H]
    final_gold = linear_down_gold  # [B, N, H]

    # ===================== Expert Selection Stream =====================
    print(f"Expert selection: \n{routing_expert_selection}")
    control = SelectGen(
        is_multihot=True,
        tensor=routing_expert_selection,
    )
    # [1, B, N]
    print(f"Control stream shape: {control.stream.shape}")

    # ===================== Tiling Config =====================
    gate_up_tile_config = LinearTileConfig(m=1, n=32, k=32)
    down_tile_config = LinearTileConfig(m=1, n=32, k=32)

    # ===================== FLOP Allocation =====================
    GATE_COMPUTE_BW = 1022
    UP_COMPUTE_BW = 1022
    ACT_FN_COMPUTE_BW = 1022
    MULT_COMPUTE_BW = 1022
    DOWN_COMPUTE_BW = 1022

    # ===================== Create Step Graph =====================
    # ---------- Formatting Inputs ----------
    # Input
    step_graph: MultiDiGraph = MultiDiGraph()

    in_load = OffChipLoad(
        underlying=input_tensor,  # [B, N, H] => [B, N // 1, H // 32]
        stride=(
            N // gate_up_tile_config.m * H // gate_up_tile_config.k,
            H // gate_up_tile_config.k,
            1,
        ),
        out_shape_tiled=(
            B,
            N // gate_up_tile_config.m,
            H // gate_up_tile_config.k,
        ),
        tile_row=gate_up_tile_config.m,
        tile_col=gate_up_tile_config.k,
        par_dispatch=4,
    )  # [1, B, N, H // 32]

    buff_input = Bufferize(
        graph=step_graph,
        input=in_load,
        rank=1,
    )  # [1, B, N] | [H // 32]

    formatted_input = Streamify(
        graph=step_graph,
        input=buff_input,
        repeat_factor=[MLP_HID // gate_up_tile_config.n],
        rank=1,
    )  # [1, B, N, MLP_HID // 32, H // 32]

    # W_gate
    w_gate_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders_gate = []
    for i in range(EXPERT):
        expert_loaders_gate.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_gate_partition, i),  # [Di]
                underlying=w_gate_list[
                    i
                ],  # [H, MLP_HID] => [H // gate_up_tile_config.k, MLP_HID // gate_up_tile_config.n]
                stride=(1, MLP_HID // gate_up_tile_config.n),
                out_shape_tiled=(
                    MLP_HID // gate_up_tile_config.n,
                    H // gate_up_tile_config.k,
                ),
                tile_row=gate_up_tile_config.k,
                tile_col=gate_up_tile_config.n,
                par_dispatch=4,
            )  # [Di, MLP_HID // 32, H // 32] = [2,4,2] x 4
        )

    REASSEMBLE_RANK = 2
    w_gate_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders_gate,
        control=control,
        reassemble_rank=REASSEMBLE_RANK,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )  # [1, B, N, 1, MLP_HID // 32, H // 32] = [1,2,4,1,4,2]
    """
    input: [2,4,2]
    ref: [1,2,4]
    output: [1,2,4,1,4,2]
    """

    # Substitute the dynamic dim in the stream as we have a statically known number of selected experts
    w_gate_reassemble.stream.shape = (
        w_gate_reassemble.stream.shape[: -(REASSEMBLE_RANK + 1)]
        + (1,)
        + w_gate_reassemble.stream.shape[-REASSEMBLE_RANK:]
    )  # [1, B, N, 1, MLP_HID // 32, H // 32]

    flattened_w_gate_reassemble = Flatten(
        graph=step_graph, input=w_gate_reassemble, min_rank=1, max_rank=2
    )  # output: [1,2,4,4,2]

    w_up_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders_up = []
    for i in range(EXPERT):
        expert_loaders_up.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_up_partition, i),  # [Di]
                underlying=w_up_list[
                    i
                ],  # [H, MLP_HID] => [H // gate_up_tile_config.k, MLP_HID // gate_up_tile_config.n]
                stride=(1, MLP_HID // gate_up_tile_config.n),
                out_shape_tiled=(
                    MLP_HID // gate_up_tile_config.n,
                    H // gate_up_tile_config.k,
                ),
                tile_row=gate_up_tile_config.k,
                tile_col=gate_up_tile_config.n,
                par_dispatch=4,
            )  # [Di, MLP_HID // 32, H // 32]
        )

    REASSEMBLE_RANK = 2
    w_up_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders_up,
        control=control,
        reassemble_rank=REASSEMBLE_RANK,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )

    # Substitute the dynamic dim in the stream as we have a statically known number of selected experts
    w_up_reassemble.stream.shape = (
        w_up_reassemble.stream.shape[: -(REASSEMBLE_RANK + 1)]
        + (1,)
        + w_up_reassemble.stream.shape[-REASSEMBLE_RANK:]
    )

    flattened_w_up_reassemble = Flatten(
        graph=step_graph, input=w_up_reassemble, min_rank=1, max_rank=2
    )

    # ---------- Computation ----------
    # Linear (Gate)
    gate = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=flattened_w_gate_reassemble,
        fn=map_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(gate_up_tile_config.m, gate_up_tile_config.n),
            dtype=formatted_input.stream.stream_dtype.tile_dtype,
        ),
        rank=1,
        write_back_mu=False,
        compute_bw=GATE_COMPUTE_BW,
    )  # [1, B, N, MLP_HID // 32]

    # Activation (SiLU)
    silu_gate = UnaryMap(
        graph=step_graph,
        input=gate,
        fn=map_fn.Silu(),
        write_back_mu=False,
        compute_bw=ACT_FN_COMPUTE_BW,
    )

    # output = OffChipStore(
    #     graph=step_graph,
    #     input=gate,  # silu_gate,
    #     par_dispatch=4,
    # )

    # Linear (Up)
    up = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=flattened_w_up_reassemble,
        fn=map_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(gate_up_tile_config.m, gate_up_tile_config.n),
            dtype=formatted_input.stream.stream_dtype.tile_dtype,
        ),
        rank=1,
        write_back_mu=False,
        compute_bw=UP_COMPUTE_BW,
    )  # [1, B, N, MLP_HID // 32]

    # Elementwise Multiplication (Gate * Up)
    gate_up = BinaryMap(
        graph=step_graph,
        in1=silu_gate,
        in2=up,
        fn=map_fn.Mul(),
        write_back_mu=True,
        compute_bw=MULT_COMPUTE_BW,
    )  # [1, B, N, MLP_HID // 32]

    # Linear (Down)
    # bufferize 1D
    buff_gate_up = Bufferize(graph=step_graph, input=gate_up, rank=1)  # [1, B, N]
    formatted_gate_up = Streamify(
        graph=step_graph,
        input=buff_gate_up,
        repeat_factor=[H // down_tile_config.n],
        rank=1,
    )  # [1, B, N, H // 32, MLP_HID // 32]

    w_down_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders_down = []
    for i in range(EXPERT):
        expert_loaders_down.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_down_partition, i),  # [Di]
                underlying=w_down_list[
                    i
                ],  # [MLP_HID, H] => [MLP_HID // down_tile_config.k, H // down_tile_config.n]
                stride=(1, H // down_tile_config.n),
                out_shape_tiled=(
                    H // down_tile_config.n,
                    MLP_HID // down_tile_config.k,
                ),
                tile_row=down_tile_config.k,
                tile_col=down_tile_config.n,
                par_dispatch=4,
            )  # [Di, MLP_HID // 32, H // 32]
        )

    REASSEMBLE_RANK = 2
    w_down_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders_down,
        control=control,
        reassemble_rank=REASSEMBLE_RANK,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )

    # Substitute the dynamic dim in the stream as we have a statically known number of selected experts
    w_down_reassemble.stream.shape = (
        w_down_reassemble.stream.shape[: -(REASSEMBLE_RANK + 1)]
        + (1,)
        + w_down_reassemble.stream.shape[-REASSEMBLE_RANK:]
    )

    flattened_w_down_reassemble = Flatten(
        graph=step_graph, input=w_down_reassemble, min_rank=1, max_rank=2
    )

    # ---------- Computation ----------
    # Linear (Down)
    down = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_gate_up,
        in2=flattened_w_down_reassemble,
        fn=map_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(down_tile_config.m, down_tile_config.n),
            dtype=formatted_gate_up.stream.stream_dtype.tile_dtype,
        ),
        rank=1,
        write_back_mu=True,
        compute_bw=DOWN_COMPUTE_BW,
    )  # [1, B, N, H // 32]

    # Output
    output = OffChipStore(
        graph=step_graph,
        input=down,
        par_dispatch=4,
    )

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Check whether the shapes match ================
    print(f"Gold shape: {final_gold.shape}")
    print(f"Output shape: {output.get_untiled_shape()}")

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "input_stationary"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )

    check_gold_tensor("output", final_gold)
