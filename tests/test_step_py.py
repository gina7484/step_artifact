import torch
from step_py.ops import OffChipLoad, OffChipStore, BinaryMap, RepeatStatic
from step_py.functions import map_fn
from sim import simulate, HBMConfig
from utils.shape_checking import is_valid_view
from utils.gold_checking import check_gold_tensor
from utils.draw_graph import save_graph_format
from rewrite.broadcast import infer_broadcast
from networkx import MultiDiGraph, drawing

# ================ Setting up the model and data ================
B = 32
H = 64

model = torch.nn.Linear(H, H, bias=False)
input = torch.randn(B, H)
weight = model.weight.T.detach().clone().contiguous()
gold = torch.matmul(input, weight)


# ================ Generating Tiling Schedule ================
tile_m_gen_q = 16
tile_k_gen_q = H
tile_n_gen_q = 32

weight_stride = (
    (0, H // tile_n_gen_q, 1)
    if H // tile_k_gen_q == 1 or H // tile_n_gen_q == 1
    else (0, 1, H // tile_n_gen_q)
)

# ================ STeP Program ================

# step_graph = []
step_graph: MultiDiGraph = MultiDiGraph()

input_stream = OffChipLoad(
    # graph=step_graph,
    underlying=input,
    stride=(H // tile_k_gen_q, 1),
    out_shape_tiled=(B // tile_m_gen_q, H // tile_k_gen_q),
    tile_row=tile_m_gen_q,
    tile_col=tile_k_gen_q,
    par_dispatch=4,
)
# print(f"input_stream shape: {input_stream.stream.shape}\n")

repeat_input_stream = RepeatStatic(
    graph=step_graph, input=input_stream, repeat_factor=H // tile_n_gen_q
)
# print(f"repeat_input_stream shape: {repeat_input_stream.stream.shape}\n")


weight_stream = OffChipLoad(
    # graph=step_graph,
    underlying=weight,
    stride=weight_stride,
    out_shape_tiled=(B // tile_m_gen_q, H // tile_k_gen_q, H // tile_n_gen_q),
    tile_row=tile_k_gen_q,
    tile_col=tile_n_gen_q,
    par_dispatch=4,
)
# print(f"weight_stream shape: {weight_stream.stream.shape}\n")


matmul = BinaryMap(
    step_graph, repeat_input_stream, weight_stream, map_fn.Matmul(), True, 1022
)
# print(f"matmul shape: {matmul.stream.shape}\n")


output_stream1 = OffChipStore(
    graph=step_graph,
    input=matmul,
    par_dispatch=4,
    store_file_name="output1",
)

output_stream2 = OffChipStore(
    graph=step_graph,
    input=matmul,
    par_dispatch=4,
    store_file_name="output2",
)

# ================ Check whether the shapes match ================

assert is_valid_view(
    gold,
    output_stream1.get_untiled_shape(),
)
print("The stream shapes match")

# ================ Rewrite passes ================
step_graph = infer_broadcast(step_graph)

# ================ Print the STeP Graph ================
# print([str(op) for op in list(step_graph.nodes(data=True))])
output_filename = f"output_step"

# drawing.nx_pydot.write_dot(step_graph, "graph.dot")
save_graph_format(step_graph, output_filename, ["png"])


# # ================ Simulate & Check the output ================
simulate(
    step_graph,
    False,  # logging
    HBMConfig(64, 8, 2, 2, 1, 14),
    "/scratch/zgh23/step_tl/graph.pb"
)

check_gold_tensor(
    "output1",
    gold.reshape(
        2, 16, 64
    ),  # Reshaping the gold because we're not using a flatten after map
    # Once we add a flatten after map, we can remove the reshape here
)

check_gold_tensor(
    "output2",
    gold.reshape(
        2, 16, 64
    ),  # Reshaping the gold because we're not using a flatten after map
    # Once we add a flatten after map, we can remove the reshape here
)
