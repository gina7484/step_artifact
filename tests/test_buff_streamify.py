import torch
from step_py.ops import *
from step_py.functions import map_fn
from sim import simulate, HBMConfig
from utils.shape_checking import is_valid_view
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from networkx import MultiDiGraph, drawing


def test_buff_streamify():
    # ================ Setting up the model and data ================
    M = 32
    K = 48

    tile_M = 16
    tile_K = 16

    input = torch.randn(M, K)
    gold = input

    # # ================ STeP Program ================

    step_graph: MultiDiGraph = MultiDiGraph()

    load = OffChipLoad(
        underlying=input,
        stride=(K // tile_K, 1),
        out_shape_tiled=(M // tile_M, K // tile_K),
        tile_row=tile_M,
        tile_col=tile_K,
        par_dispatch=4,
    )

    buff = Bufferize(step_graph, load, 1)

    streamify = Streamify(step_graph, buff, [], 1)

    output = OffChipStore(
        graph=step_graph,
        input=streamify,
        par_dispatch=4,
        store_file_name="output",
    )

    # ================ Check whether the shapes match ================

    print(f"Gold shape: {gold.shape}")
    print(f"Output shape: {output.get_untiled_shape()}")

    # ================ Rewrite passes ================
    step_graph = infer_broadcast(step_graph)

    # ================ Print the STeP Graph ================
    OUTPUT_FILENAME = "buff_streamify_step"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])

    # ================ Simulate & Check the output ================
    simulate(
        step_graph,
        False,  # logging
        HBMConfig(64, 8, 2, 2, 1, 14),
        "/home/ginasohn/step_tl/graph.pb",
    )

    check_gold_tensor("output", gold)


def test_buff_dyn_streamify():
    pass
