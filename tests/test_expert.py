from networkx import MultiDiGraph
import torch
from step_py.kernels.linear import LinearTileConfig
from step_py.utility_ops import *
from step_py.ops import *
import numpy as np
from sim import simulate, HBMConfig


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


def test_expert_mnk_mnk():
    B = 32
    H = 64
    MLP_HID = 128

    gate_tile_config = LinearTileConfig(m=1, n=16, k=16)
    up_tile_config = LinearTileConfig(m=1, n=16, k=16)
    down_tile_config = LinearTileConfig(m=1, n=16, k=16)

    input = torch.randn(B, H)

    _, expert_selection = torch.topk(torch.randn(B, H), 2, dim=-1)

    # pylint: disable=not-callable
    expert_selection_one_hot = torch.nn.functional.one_hot(expert_selection)
    expert_selection_multi_hot = expert_selection_one_hot.sum(dim=-2)
