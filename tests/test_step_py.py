import torch
from step_py.ops import OffChipLoad, OffChipStore, BinaryMap, RepeatStatic
from step_py.functions import map_fn

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

input_stream = OffChipLoad(
    underlying=input,
    stride=(H // tile_k_gen_q, 1),
    out_shape_tiled=(B // tile_m_gen_q, H // tile_k_gen_q),
    tile_row=tile_m_gen_q,
    tile_col=tile_k_gen_q,
)
repeat_input_stream = RepeatStatic(input=input_stream, repeat_factor=H // tile_n_gen_q)

weight_stream = OffChipLoad(
    underlying=weight,
    stride=weight_stride,
    out_shape_tiled=(B // tile_m_gen_q, H // tile_k_gen_q, H // tile_n_gen_q),
    tile_row=tile_k_gen_q,
    tile_col=tile_n_gen_q,
)

matmul = BinaryMap(repeat_input_stream, weight_stream, map_fn.Matmul(), True, 1022)

output_stream = OffChipStore(
    tensor_shape_tiled=(B // tile_m_gen_q, H // tile_n_gen_q),
    tile_row=tile_m_gen_q,
    tile_col=tile_n_gen_q,
)

assert output_stream.get_untiled_shape() == tuple(gold.shape)
print(
    f"Passed! Output Stream({output_stream.get_untiled_shape()}) == Gold Shape({tuple(gold.shape)})"
)
