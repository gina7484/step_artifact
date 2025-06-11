from networkx import MultiDiGraph
import torch
from step_py.kernels.linear import LinearTileConfig
from step_py.utility_ops import *
from step_py.ops import *
import numpy as np
from sim import simulate, HBMConfig
from step_py.functions import map_fn
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast


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
    N = 16
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

    _, expert_selection = torch.topk(torch.randn(B, H), 2, dim=-1)

    # pylint: disable=not-callable
    expert_selection_one_hot = torch.nn.functional.one_hot(expert_selection)
    expert_selection_multi_hot = expert_selection_one_hot.sum(dim=-2)

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
    w_gate_load = DynOffChipLoad(
        graph=step_graph,
        ref=buff_input,
        underlying=w_gate,  # [H, MLP_HID] => [H // gate_up_tile_config.k, MLP_HID // gate_up_tile_config.n]
        stride=(1, MLP_HID // gate_up_tile_config.n),
        out_shape_tiled=(
            MLP_HID // gate_up_tile_config.n,
            H // gate_up_tile_config.k,
        ),
        tile_row=gate_up_tile_config.k,
        tile_col=gate_up_tile_config.n,
        par_dispatch=4,
    )  # [1, B, N, MLP_HID // 32, H // 32]

    # W_up
    w_up_load = DynOffChipLoad(
        graph=step_graph,
        ref=buff_input,
        underlying=w_up,  # [H, MLP_HID] => [H // gate_up_tile_config.k, MLP_HID // gate_up_tile_config.n]
        stride=(1, MLP_HID // gate_up_tile_config.n),
        out_shape_tiled=(
            MLP_HID // gate_up_tile_config.n,
            H // gate_up_tile_config.k,
        ),
        tile_row=gate_up_tile_config.k,
        tile_col=gate_up_tile_config.n,
        par_dispatch=4,
    )  # [1, B, N, MLP_HID // 32, H // 32]

    # ---------- Computation ----------
    # Linear (Gate)
    gate = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=w_gate_load,
        fn=map_fn.Matmul(),
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
        in2=w_up_load,
        fn=map_fn.Matmul(),
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

    w_down_load = DynOffChipLoad(
        graph=step_graph,
        ref=buff_input,
        underlying=w_down,  # [MLP_HID, H] => [MLP_HID // down_tile_config.k, H // down_tile_config.n]
        stride=(1, H // down_tile_config.n),
        out_shape_tiled=(
            H // down_tile_config.n,
            MLP_HID // down_tile_config.k,
        ),
        tile_row=down_tile_config.k,
        tile_col=down_tile_config.n,
        par_dispatch=4,
    )  # [1, B, N, H // 32, MLP_HID // 32]

    down = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_gate_up,
        in2=w_down_load,
        fn=map_fn.Matmul(),
        rank=1,
        write_back_mu=True,
        compute_bw=DOWN_COMPUTE_BW,
    )  # [1, B, N, H // 32]

    # Output
    output_up = OffChipStore(
        graph=step_graph,
        input=down,
        par_dispatch=4,
    )
    # ================ Check whether the shapes match ================

    # print(f"Gold (gate) shape: {gold_gate.shape}")
    # print(f"Output shape: {output_gate.get_untiled_shape()}")

    # print(f"Gold (up) shape: {gold_up.shape}")
    # print(f"Output shape: {output_up.get_untiled_shape()}")

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    print(f"Output shape: {output_up.get_untiled_shape()}")
    print(f"{[B,N,H]}")

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "input_stationary"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    # ================ Simulate & Check the output ================
    # simulate(
    #     step_graph,
    #     False,  # logging
    #     HBMConfig(64, 8, 2, 2, 1, 14),
    #     "/home/ginasohn/step_tl/graph.pb",
    # )

    # check_gold_tensor("output", gold_gate)
    # check_gold_tensor("up_output", gold_up)
