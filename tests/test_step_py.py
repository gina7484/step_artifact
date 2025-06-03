import torch
from step_py.ops import OffChipLoad, OffChipStore, BinaryMap, RepeatStatic
from step_py.functions import map_fn
from sim import simulate, HBMConfig
from utils.shape_checking import is_valid_view
from utils.gold_checking import check_gold_tensor

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

# Combined operation

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
    store_file_name="output",
)

# ================ Check whether the shapes match ================

assert is_valid_view(
    gold,
    output_stream.get_untiled_shape(),
)
print("The stream shapes match")


# ================ Print the STeP Graph ================
print([str(op) for op in step_graph])
simulate(
    step_graph,
    False,
    HBMConfig(64, 8, 4, 4, 1, 14),
)

# ================ Check the output ================
check_gold_tensor(
    "output",
    gold.reshape(
        2, 16, 64
    ),  # Reshaping the gold because we're not using a flatten after map
    # Once we add a flatten after map, we can remove the reshape here
)
