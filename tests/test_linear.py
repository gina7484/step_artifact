import torch
from step_py.kernels.linear import Linear, LinearTileConfig
from step_py.ops import *
from step_py.functions import map_fn
from sim import simulate, HBMConfig
from utils.shape_checking import is_valid_view
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from networkx import MultiDiGraph, drawing

# ================ Setting up the model and data ================
M = 32
N = 64
K = 48

model = torch.nn.Linear(K, N, bias=False)
input = torch.randn(M, K)
# input = torch.ones(M, K, dtype=torch.float32)
weight = model.weight.T.detach().clone().contiguous()  # [K,N]
gold = model(input)
# weight = torch.ones(K, N, dtype=torch.float32)  # [K,N]
# gold = torch.matmul(input, weight)  # [M,N]

# # ================ STeP Program ================

step_graph: MultiDiGraph = MultiDiGraph()

linear = Linear(
    step_graph=step_graph,
    input=input,
    weight=weight,
    tile_config=LinearTileConfig(m=16, k=16, n=N),
    comp_bw=1024,
    write_back_mu=True,
    par_dispatch=4,
)

output = OffChipStore(
    graph=step_graph,
    input=linear,
    par_dispatch=4,
    store_file_name="output",
)

# ================ Check whether the shapes match ================

print(f"Gold shape: {gold.shape}")
print(f"Output shape: {output.get_untiled_shape()}")

# ================ Rewrite passes ================
step_graph = infer_broadcast(step_graph)

# ================ Print the STeP Graph ================
OUTPUT_FILENAME = "linear_tile_mn_step"
save_graph_format(step_graph, OUTPUT_FILENAME, ["png"])


# ================ Simulate & Check the output ================
simulate(
    step_graph,
    False,  # logging
    HBMConfig(64, 8, 2, 2, 1, 14),
    "/home/ginasohn/step_tl/graph.pb",
)

check_gold_tensor("output", gold)
