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
    print(expert_selection_multi_hot)
    select_gan = SelectGen(
        is_multihot=True,
        tensor=expert_selection_multi_hot,
    )

    printer_context = PrinterContext(step_graph, select_gan)

    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/scratch/zgh23/step_tl/graph.pb",
    )
