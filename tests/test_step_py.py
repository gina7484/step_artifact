import torch
from step_py.ops import OffChipLoad, OffChipStore, BinaryMap, RepeatStatic
from step_py.functions import map_fn
from sim import simulate
from utils.shape_checking import is_valid_view


# ================ Setting up the model and data ================
B = 32
H = 64

model = torch.nn.Linear(H, H)

input = torch.randn(B, H)
gold = model(input)

weight = model.weight

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

step_graph = []

input_stream = OffChipLoad(
    graph=step_graph,
    underlying=input,
    stride=(H // tile_k_gen_q, 1),
    out_shape_tiled=(B // tile_m_gen_q, H // tile_k_gen_q),
    tile_row=tile_m_gen_q,
    tile_col=tile_k_gen_q,
)

repeat_input_stream = RepeatStatic(
    graph=step_graph, input=input_stream, repeat_factor=H // tile_n_gen_q
)

weight_stream = OffChipLoad(
    graph=step_graph,
    underlying=weight,
    stride=weight_stride,
    out_shape_tiled=(B // tile_m_gen_q, H // tile_k_gen_q, H // tile_n_gen_q),
    tile_row=tile_k_gen_q,
    tile_col=tile_n_gen_q,
)


matmul = BinaryMap(
    step_graph, repeat_input_stream, weight_stream, map_fn.Matmul(), True, 1022
)


output_stream = OffChipStore(
    graph=step_graph,
    input=matmul,
)

# ================ Check whether the shapes match ================

assert is_valid_view(gold, output_stream.get_untiled_shape())

print(
    f"Passed! Gold Shape({tuple(gold.shape)}) can be viewed as Output Stream({output_stream.get_untiled_shape()}) "
)

# ================ Print the STeP Graph ================
print([str(op) for op in step_graph])
simulate(step_graph)
