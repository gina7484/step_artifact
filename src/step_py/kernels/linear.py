from dataclasses import dataclass
from abc import ABC
from typing import Union, Tuple

from networkx import MultiDiGraph
import torch

from step_py.datatype import Stream
from step_py.functions import map_fn
from step_py.ops import *


@dataclass
class LinearTileConfig:
    m: int
    k: int
    n: int


def Linear(
    step_graph: MultiDiGraph,
    input: Union[torch.Tensor, StepOps, Tuple[StepOps, int]],
    weight: torch.Tensor,
    tile_config: LinearTileConfig,
    comp_bw: int,
    write_back_mu: bool,
    par_dispatch: int = 4,
) -> StepOps:
    # ================= (Load) & Format the input stream =================
    formatted_input = None
    outer_dims = ()

    weight_tensor_shape = tuple(weight.shape)
    assert len(weight_tensor_shape) == 2
    K = weight_tensor_shape[0]
    N = weight_tensor_shape[1]

    if isinstance(input, torch.Tensor):
        # Loading from off-chip
        input_tensor_shape = tuple(input.shape)
        assert len(input_tensor_shape) >= 2
        formatted_input = OffChipLoad(
            underlying=input,
            stride=(K // tile_config.k, 0, 1),
            out_shape_tiled=input_tensor_shape[:-2]
            + (
                input_tensor_shape[-2] // tile_config.m,
                N // tile_config.n,
                input_tensor_shape[-1] // tile_config.k,
            ),
            tile_row=tile_config.m,
            tile_col=tile_config.k,
            par_dispatch=par_dispatch,
        )
        outer_dims = input_tensor_shape[:-2] + (
            input_tensor_shape[-2] // tile_config.m,
        )

        print(f"Input shape: {formatted_input.stream.shape}")
    elif isinstance(input, (StepOps, Tuple)):
        if N == tile_config.n:  # TileM / TileMK (Not tiling N)
            formatted_input = Promote(graph=step_graph, input=input, promote_rank=1)
        else:  # TileMN / TileMNK (N is tiled)
            """
            Bufferize 1D
            Streamify with Repeat 1D
            """
            pass

    # ================= Load weight =================
    formatted_weight = OffChipLoad(
        underlying=weight,
        stride=(0,) * len(outer_dims) + (1, N // tile_config.n),
        out_shape_tiled=outer_dims
        + (
            N // tile_config.n,
            K // tile_config.k,
        ),
        tile_row=tile_config.k,
        tile_col=tile_config.n,
        par_dispatch=par_dispatch,
    )
    print(f"Weight shape: {formatted_weight.stream.shape}")

    # ================= Computation =================
    result = BinaryMapAccum(
        step_graph,
        formatted_input,
        formatted_weight,
        map_fn.Matmul(),
        1,
        write_back_mu,
        comp_bw,
    )

    return result
