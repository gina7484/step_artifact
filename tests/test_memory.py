from networkx import MultiDiGraph
import torch

from step_py.ops import *
from step_py.utility_ops import *
from sim import HBMConfig, SimConfig, simulate
from utils.gold_checking import check_gold_tensor


def test_random_offchip_store():

    step_graph = MultiDiGraph()

    B = 16
    D = 128
    tensor = torch.zeros((4 * B, D), dtype=torch.float32)

    req0 = torch.randn(4, D)
    req3 = torch.randn(4, D)
    req11 = torch.randn(4, D)

    wdata = OffChipLoad(
        underlying=torch.cat([req0, req3, req11], dim=0),  # Shape: [12, D]
        stride=(1, 1),
        out_shape_tiled=(3, 1),
        tile_row=4,
        tile_col=D,
        par_dispatch=1,
    )  # [1,3,1]

    waddr = MetadataGen(tensor=torch.tensor([0, 3, 11], dtype=torch.uint64))  # [1,3]

    store = RandomOffChipStore(
        graph=step_graph,
        underlying=tensor,
        wdata=wdata,
        waddr=waddr,
        tile_row=4,
        tile_col=D,
        base_addr_byte=0,
        par_dispatch=1,
    )

    consumer = ConsumerContext(graph=step_graph, input=store)

    hbm_config = HBMConfig(64, 32, 2, 2, 1, 14)
    sim_config = SimConfig(channel_depth=16, functional_sim=True, mock_bf16=False)
    cycles, duration_ms, duration_s = simulate(
        step_graph,
        False,  # logging
        hbm_config,
        sim_config,
        "/home/ginasohn/step_tl/graph.pb",
    )

    for idx, req in zip([0, 3, 11], [req0, req3, req11]):
        tensor[idx * 4 : (idx + 1) * 4, :] = req

    check_gold_tensor(store.store_file_name, tensor.contiguous())
