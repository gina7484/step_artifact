import torch
from step_py.ops import *
from step_py.utility_ops import *
from step_py.functions import map_fn
from sim import simulate, HBMConfig
from utils.shape_checking import is_valid_view
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from networkx import MultiDiGraph, drawing


def test_prefill_expert_mnk_mnk():
    # ===================== Input Creation =====================
    B = 16
    N = 64
    H = 128

    input_tensor = torch.randn(B, N, H)

    step_graph: MultiDiGraph = MultiDiGraph()

    input = OffChipLoad(
        underlying=input_tensor,  # [B, N, H] => [B, N // 1, H // 32]
        stride=(
            N // 1 * H // 32,
            H // 32,
            1,
        ),
        out_shape_tiled=(
            B,
            N // 1,
            H // 32,
        ),
        tile_row=1,
        tile_col=32,
        par_dispatch=4,
    )  # [1, B, N // 1, H // 32]

    print = OffChipStore(
        graph=step_graph,
        input=input,
        par_dispatch=4,
    )

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "linear_gate_up"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    # ================ Simulate & Check the output ================
    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )

    check_gold_tensor("output", input_tensor)
