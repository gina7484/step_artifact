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
from utils.moe import moe_linear_gold_calc_batched


def test_0d_partition_onehot():

    # ===================== Data Creation =====================
    B = 2
    N = 3
    EXPERT = 4  # number of experts
    H = 64
    MLP_HID = 128

    # ------------ Selection ------------
    expert_indices = torch.tensor([[0, 1, 2], [1, 1, 2]])  # [B,N]
    # pylint: disable=not-callable
    routing_expert_selection = torch.nn.functional.one_hot(
        expert_indices, num_classes=EXPERT
    )  # [B, N, EXPERT]

    # ------------ Input ------------
    input_tensor = torch.randn(B, N, H)

    # ------------ Expert Weights ------------

    # ------------ Gold calculation ------------

    # ===================== Expert Selection Stream =====================
    print(f"Expert selection: \n{routing_expert_selection}")
    control = SelectGen(is_multihot=True, tensor=routing_expert_selection, n=EXPERT)
    print(f"Control stream shape: {control.stream.shape}")

    # ===================== Tiling Config =====================

    step_graph: MultiDiGraph = MultiDiGraph()

    w_gate_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert0 = ConsumerContext(graph=step_graph, input=(w_gate_partition, 0))
    expert1 = ConsumerContext(graph=step_graph, input=(w_gate_partition, 1))
    expert2 = ConsumerContext(graph=step_graph, input=(w_gate_partition, 2))
    expert3 = PrinterContext(graph=step_graph, input=(w_gate_partition, 3))

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "test_partition_2_experts"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )


def test_0d_partition_multihot():

    # ===================== Data Creation =====================
    B = 2
    N = 3
    EXPERT = 4  # number of experts
    H = 64
    MLP_HID = 128

    # ------------ Selection ------------
    expert_indices = torch.tensor(
        [[[0, 1], [1, 2], [1, 2]], [[1, 0], [2, 1], [2, 3]]]
    )  # [B,N,2]
    # pylint: disable=not-callable
    routing_expert_selection = torch.nn.functional.one_hot(
        expert_indices, num_classes=EXPERT
    )  # [B, N, 2, EXPERT]
    expert_selection_multi_hot = routing_expert_selection.sum(dim=-2)  # [B, N, EXPERT]

    # ------------ Input ------------
    input_tensor = torch.randn(B, N, H)

    # ------------ Expert Weights ------------

    # ------------ Gold calculation ------------

    # ===================== Expert Selection Stream =====================
    print(f"Expert selection: \n{expert_selection_multi_hot}")
    control = SelectGen(is_multihot=True, tensor=expert_selection_multi_hot, n=EXPERT)
    print(f"Control stream shape: {control.stream.shape}")

    # ===================== Tiling Config =====================

    step_graph: MultiDiGraph = MultiDiGraph()

    w_gate_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert0 = ConsumerContext(graph=step_graph, input=(w_gate_partition, 0))
    expert1 = PrinterContext(graph=step_graph, input=(w_gate_partition, 1))
    expert2 = ConsumerContext(graph=step_graph, input=(w_gate_partition, 2))
    expert3 = ConsumerContext(graph=step_graph, input=(w_gate_partition, 3))

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "test_partition_2_experts"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )


def test_0d_partition_onehot_dyn_load():

    # ===================== Data Creation =====================
    B = 2
    N = 3
    EXPERT = 4  # number of experts
    H = 64
    MLP_HID = 128

    # ------------ Selection ------------
    expert_indices = torch.tensor([[0, 1, 2], [3, 1, 2]])  # [B,N]
    # pylint: disable=not-callable
    routing_expert_selection = torch.nn.functional.one_hot(
        expert_indices, num_classes=EXPERT
    )  # [B, N, EXPERT]

    # ------------ Input ------------
    input_tensor = torch.randn(B, N, H)

    # ------------ Expert Weights ------------
    expert_weights = [torch.full((H, MLP_HID), float(i)) for i in range(EXPERT)]

    # ------------ Gold calculation ------------

    # ===================== Expert Selection Stream =====================
    print(f"Expert selection: \n{routing_expert_selection}")
    control = SelectGen(
        is_multihot=True,
        tensor=routing_expert_selection,
        n=EXPERT,
    )
    # print(f"Control stream shape: {control.stream.shape}")

    # # ===================== Tiling Config =====================
    gate_up_tile_config = LinearTileConfig(m=1, n=32, k=32)

    step_graph: MultiDiGraph = MultiDiGraph()

    w_gate_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders = []
    for i in range(EXPERT):
        expert_loaders.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_gate_partition, i),  # [Di]
                underlying=expert_weights[
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

    expert0 = ConsumerContext(graph=step_graph, input=expert_loaders[0])  # [1,4,2]
    expert1 = ConsumerContext(graph=step_graph, input=expert_loaders[1])  # [2,4,2]
    expert2 = PrinterContext(graph=step_graph, input=expert_loaders[2])  # [2,4,2]
    expert3 = ConsumerContext(graph=step_graph, input=expert_loaders[3])  # [1,4,2]

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "test_0d_partition_onehot_dyn_load"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )


def test_0d_partition_onehot_dyn_load_reassemble():

    # ===================== Data Creation =====================
    B = 2
    N = 3
    EXPERT = 4  # number of experts
    H = 64
    MLP_HID = 128

    # ------------ Selection ------------
    expert_indices = torch.tensor([[0, 1, 2], [3, 1, 2]])  # [B,N]
    # pylint: disable=not-callable
    routing_expert_selection = torch.nn.functional.one_hot(
        expert_indices, num_classes=EXPERT
    )  # [B, N, EXPERT]

    # ------------ Input ------------
    input_tensor = torch.randn(B, N, H)

    # ------------ Expert Weights ------------
    expert_weights = [torch.full((H, MLP_HID), float(i)) for i in range(EXPERT)]

    # ------------ Gold calculation ------------

    # ===================== Expert Selection Stream =====================
    print(f"Expert selection: \n{routing_expert_selection}")
    control = SelectGen(is_multihot=True, tensor=routing_expert_selection, n=EXPERT)
    print(f"Control stream shape: {control.stream.shape}")  # [B,N]

    # ===================== Tiling Config =====================
    gate_up_tile_config = LinearTileConfig(m=1, n=32, k=32)

    step_graph: MultiDiGraph = MultiDiGraph()

    w_gate_partition = FlatPartition(
        graph=step_graph,
        input=control,
        control=control,
        partition_rank=0,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
        num_consumers=EXPERT,
    )  # [D0], [D1], .. [D(EXPERT-1)]

    expert_loaders = []
    for i in range(EXPERT):
        expert_loaders.append(
            DynOffChipLoad(
                graph=step_graph,
                ref=(w_gate_partition, i),  # [Di]
                underlying=expert_weights[
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

    # Input to FlatReassemble: [1,4,2] [2,4,2] [2,4,2] [1,4,2]
    # Control stream: [2,3]

    flat_reassemble = FlatReassemble(
        graph=step_graph,
        inputs=expert_loaders,
        control=control,
        reassemble_rank=2,
        switch_cycles=[1] * EXPERT,
        write_back_mu=False,
    )

    result = ConsumerContext(graph=step_graph, input=flat_reassemble)  # [2,3, 1, 4,2]

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "test_0d_partition_onehot_dyn_load_reassemble"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )
