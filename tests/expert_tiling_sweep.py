import torch
from dataclasses import dataclass
from typing import Tuple
from networkx import MultiDiGraph

from rewrite.broadcast import infer_broadcast
from sim import HBMConfig, SimConfig, simulate
from step_py.functions import init_fn, map_fn, map_accum_fn
from step_py.kernels.linear import LinearTileConfig
from step_py.ops import *
from step_py.datatype import Stream
from step_py.utility_ops import *
from utils.draw_graph import save_graph_format
from utils.gold_checking import check_gold_tensor


def gold_calc(
    input: torch.Tensor,
    w_gate: torch.nn.Module,
    w_up: torch.nn.Module,
    w_down: torch.nn.Module,
):
    gate = w_gate(input)
    up = w_up(input)
    # down = w_down(up)
    return (gate, up)


@dataclass
class DeepSeekV316B:
    n_expert_sim = 1
    # n_routed_experts = 64
    # n_activated_experts = 6
    dim = 2048
    moe_inter_dim = 1408


@dataclass
class SmallerDeepSeekV3:
    n_expert_sim = 1
    # n_routed_experts = 64
    # n_activated_experts = 6
    dim = 64  # 2048 // 32 = 64
    moe_inter_dim = 352  # 1408 // 4 (Can use tile size of 32)


@dataclass
class SmallerMixtral:  # 32x scaled down version for each dimension
    n_expert_sim = 1
    # n_routed_experts = 8
    # n_activated_experts = 2
    dim = 128  # 4096/32
    moe_inter_dim = 448  # 14336/32 (Can use tile size of 64)


@dataclass
class TinyExample:  # 32x scaled down version for each dimension
    n_expert_sim = 1
    dim = 64
    moe_inter_dim = 32


@dataclass
class Mixtral8x7b:
    n_expert_sim = 1
    # n_routed_experts = 8
    # n_activated_experts = 2
    dim = 4096
    moe_inter_dim = 14336


@dataclass
class TilingSchedule:
    gate: str
    down: str


def create_gate_up_down_ops(
    step_graph: MultiDiGraph,
    input: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    tile_config: LinearTileConfig,
    par_dispatch: int,
    write_back_mu: bool,
    comp_bw: int,
) -> Tuple[StepOps, StepOps]:

    # ================= (Load) & Format the input stream =================
    formatted_input = None
    outer_dims = ()

    assert w_gate.shape == w_up.shape
    weight_tensor_shape = tuple(w_gate.shape)
    assert len(weight_tensor_shape) == 2
    K = weight_tensor_shape[0]
    N = weight_tensor_shape[1]

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
    outer_dims = input_tensor_shape[:-2] + (input_tensor_shape[-2] // tile_config.m,)

    # ================= Load weight =================

    formatted_weight_gate = OffChipLoad(
        underlying=w_gate,
        stride=(0,) * len(outer_dims) + (1, N // tile_config.n),
        out_shape_tiled=outer_dims  # type:ignore
        + (
            N // tile_config.n,
            K // tile_config.k,
        ),
        tile_row=tile_config.k,
        tile_col=tile_config.n,
        par_dispatch=par_dispatch,
    )
    print(f"Weight (gate) shape: {formatted_weight_gate.stream.shape}")

    formatted_weight_up = OffChipLoad(
        underlying=w_up,
        stride=(0,) * len(outer_dims) + (1, N // tile_config.n),
        out_shape_tiled=outer_dims  # type:ignore
        + (
            N // tile_config.n,
            K // tile_config.k,
        ),
        tile_row=tile_config.k,
        tile_col=tile_config.n,
        par_dispatch=par_dispatch,
    )
    print(f"Weight (up) shape: {formatted_weight_up.stream.shape}")

    # ================= Computation =================
    result_gate = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=formatted_weight_gate,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(tile_config.m, tile_config.n),
            dtype=Float32(),
        ),
        rank=1,
        write_back_mu=write_back_mu,
        compute_bw=comp_bw,
    )

    result_up = BinaryMapAccum(
        graph=step_graph,
        in1=formatted_input,
        in2=formatted_weight_up,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(tile_config.m, tile_config.n),
            dtype=Float32(),
        ),
        rank=1,
        write_back_mu=write_back_mu,
        compute_bw=comp_bw,
    )

    return (result_gate, result_up)


def test_expert_tiling_sweep():
    # ------------ Sim Conig ------------
    simulate_mode = "functional"
    # simulate_mode = "timing"
    # simulate_mode = None

    check_gold = True

    logging = None

    par_dispatch = 4

    # ------------ Model Configuration ------------
    model_config = TinyExample()

    # ------------ Batch Size ------------
    B = 32

    # ------------ Compute Bandwidths ------------
    GATE_COMPUTE_BW = 1022
    UP_COMPUTE_BW = 1022
    ACT_FN_COMPUTE_BW = 1022
    MULT_COMPUTE_BW = 1022
    DOWN_COMPUTE_BW = 1022
    WEIGHT_SCALE_COMPUTE_BW = 1022
    ACCUM_COMPUTE_BW = 1022

    torch.manual_seed(42)

    # ------------ Input generation ------------
    input_tensor = torch.randn(B, model_config.dim)

    # ------------ Expert Weights (gate, up, down) ------------
    linear_gate_list = [
        torch.nn.Linear(model_config.dim, model_config.moe_inter_dim, bias=False)
        for _ in range(model_config.n_expert_sim)
    ]
    linear_up_list = [
        torch.nn.Linear(model_config.dim, model_config.moe_inter_dim, bias=False)
        for _ in range(model_config.n_expert_sim)
    ]
    linear_down_list = [
        torch.nn.Linear(model_config.moe_inter_dim, model_config.dim, bias=False)
        for _ in range(model_config.n_expert_sim)
    ]

    w_gate_list = [
        linear_gate.weight.T.detach().clone().contiguous()
        for linear_gate in linear_gate_list
    ]
    w_up_list = [
        linear_up.weight.T.detach().clone().contiguous() for linear_up in linear_up_list
    ]
    w_down_list = [
        linear_down.weight.T.detach().clone().contiguous()
        for linear_down in linear_down_list
    ]

    # ------------ Tiling Schedule ------------
    tiling_schedule_list = {
        "mkn_mk": TilingSchedule(gate="mkn", down="mk"),
        "mkn_mkn": TilingSchedule(gate="mkn", down="mkn"),
        "mk_m": TilingSchedule(gate="mk", down="m"),
        "mk_mn": TilingSchedule(gate="mk", down="mn"),
        "mn_mk": TilingSchedule(gate="mn", down="mk"),
        "mn_mkn": TilingSchedule(gate="mn", down="mkn"),
        "m_mn": TilingSchedule(gate="m", down="mn"),
        "m_m": TilingSchedule(gate="m", down="m"),
    }

    tiling_schedule_name = "mn_mk"
    tiling_schedule = tiling_schedule_list[tiling_schedule_name]

    tile_m = B  # also tile_m_down
    tile_k = model_config.dim
    tile_n = model_config.moe_inter_dim  # also tile_k_down
    tile_n_down = model_config.dim

    if tiling_schedule.gate == "mkn":
        tile_k = 16
        tile_n = 16
    elif tiling_schedule.gate == "mk":
        tile_k = 16
    elif tiling_schedule.gate == "mn":
        tile_n = 16

    if tiling_schedule.down == "mkn" or tiling_schedule.down == "mn":
        tile_n_down = 16

    gate_up_linear_config = LinearTileConfig(m=tile_m, k=tile_k, n=tile_n)
    down_linear_config = LinearTileConfig(m=tile_m, k=tile_n, n=tile_n_down)

    # ------------ Step Graph ------------
    step_graph = MultiDiGraph()

    gate, up = create_gate_up_down_ops(
        step_graph=step_graph,
        input=input_tensor,
        w_gate=w_gate_list[0],
        w_up=w_up_list[0],
        tile_config=gate_up_linear_config,
        par_dispatch=par_dispatch,
        write_back_mu=False,
        comp_bw=GATE_COMPUTE_BW,
    )

    store_gate = OffChipStore(
        graph=step_graph, input=gate, par_dispatch=par_dispatch, store_file_name="gate"
    )

    store_up = OffChipStore(
        graph=step_graph, input=up, par_dispatch=par_dispatch, store_file_name="up"
    )

    step_graph = infer_broadcast(step_graph)

    # ------------ Print Graph ------------
    OUTPUT_FILENAME = f"expert_{tiling_schedule_name}"
    save_graph_format(step_graph, OUTPUT_FILENAME, ["svg", "png"])

    # ------------ Access-Reuse Analysis ------------
    total_off_chip_traffic = sympy.Integer(0)
    total_on_chip_requirement = sympy.Integer(0)

    for node_tuple in step_graph.nodes(data=True):
        node, data = node_tuple
        if isinstance(node, StepOps):
            total_off_chip_traffic = sympy.Add(
                total_off_chip_traffic, node.off_chip_traffic()
            )
            total_on_chip_requirement = sympy.Add(
                total_on_chip_requirement, node.on_chip_requirement()
            )
        else:
            raise ValueError(f"Node {node} in the graph is not a StepOps")

    print(f"Total on-chip requirement (bytes): {total_on_chip_requirement}")
    print(f"Total off-chip traffic (bytes): {total_off_chip_traffic}")

    # ------------ Simulate ------------
    if simulate_mode == "functional":
        hbm_config = HBMConfig(64, 8, 2, 2, 1, 14)
        sim_config = SimConfig(channel_depth=1)

        if logging is None:
            simulate(
                step_graph,
                False,  # logging
                hbm_config,
                sim_config,
                "/home/ginasohn/step_tl/graph.pb",
            )
        else:
            assert isinstance(logging, str), "Logging must be a string path"
            simulate(
                step_graph,
                True,  # logging
                hbm_config,
                sim_config,
                "/home/ginasohn/step_tl/graph.pb",
                logging,
            )

    elif simulate_mode == "timing":
        pass

    # ------------ Gold Calculation & Verification ------------

    if check_gold:
        gate, up = gold_calc(
            input=input_tensor,
            w_gate=linear_gate_list[0],
            w_up=linear_up_list[0],
            w_down=linear_down_list[0],
        )
        print(f"Gate: {store_gate.get_untiled_shape()}")
        check_gold_tensor(store_gate.store_file_name, gate)
        print(f"Up: {store_up.get_untiled_shape()}")
        check_gold_tensor(store_up.store_file_name, up)
