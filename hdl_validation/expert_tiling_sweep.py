import csv
import torch
from dataclasses import dataclass, fields
from typing import Tuple
from networkx import MultiDiGraph

from rewrite.broadcast import infer_broadcast
from sim import HBMConfig, SimConfig, simulate
from step_py.functions import init_fn, map_fn, map_accum_fn
from step_py.kernels.linear import Linear, LinearTileConfig
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
    proj = up * torch.nn.functional.silu(gate)
    down = w_down(proj)
    return down


@dataclass
class DeepSeekV316B:
    n_expert_sim = 1
    # n_routed_experts = 64
    # n_activated_experts = 6
    dim = 2048
    moe_inter_dim = 1408


@dataclass
class Qwen30b:
    n_expert_sim = 1
    # n_routed_experts = 128
    # n_activated_experts = 8
    dim = 2048
    moe_inter_dim = 768


@dataclass
class SimpleExample:  # 32x scaled down version for each dimension
    n_expert_sim = 1
    dim = 256
    moe_inter_dim = 512


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


@dataclass
class ResultMetrics:
    b: int
    dim: int
    moe_inter_dim: int
    tiling_schedule: str
    tile_m: int
    tile_k: int
    tile_n: int
    tile_n_down: int
    gate_up_compute_bw: int
    act_fn_compute_bw: int
    mult_compute_bw: int
    down_compute_bw: int
    flops: int
    simulate_mode: str
    mem_bw: int = 0  # GB/cycle
    ridge_point: float = 0  # flops/byte
    operational_intensity: float = 0  # flops/byte
    cycles: int = 0
    on_chip_requirement: int = 0
    off_chip_traffic: int = 0
    channel_depth: int = 0
    duration_ms: int = 0
    duration_s: int = 0


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


def create_gate_up_down_ops_repeat(
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
    raw_input = OffChipLoad(
        underlying=input,
        stride=(K // tile_config.k, 1),
        out_shape_tiled=input_tensor_shape[:-2]
        + (
            input_tensor_shape[-2] // tile_config.m,
            # N // tile_config.n,
            input_tensor_shape[-1] // tile_config.k,
        ),
        tile_row=tile_config.m,
        tile_col=tile_config.k,
        par_dispatch=par_dispatch,
    )
    outer_dims = input_tensor_shape[:-2] + (input_tensor_shape[-2] // tile_config.m,)

    flattened_raw_input = Flatten(
        graph=step_graph,
        input=raw_input,
        min_rank=0,
        max_rank=1,
    )

    repeated_raw_input = RepeatStatic(
        graph=step_graph, input=flattened_raw_input, repeat_factor=N // tile_config.n
    )

    formatted_input = Reshape(
        graph=step_graph,
        input=repeated_raw_input,
        chunk_size=1,
        reshape_rank=0,
        write_back_mu=False,
    )

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

    csv_filename = "expert_tiling_sweep.csv"

    # ------------ Model Configuration ------------
    model_config = SimpleExample()

    # ------------ Batch Size ------------
    B = 32

    # ------------ Compute Bandwidths ------------
    GATE_UP_COMPUTE_BW = 2048
    ACT_FN_COMPUTE_BW = 2048
    MULT_COMPUTE_BW = 2048
    DOWN_COMPUTE_BW = 2048

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
        "kn_k": TilingSchedule(gate="kn", down="k"),
        "kn_kn": TilingSchedule(gate="kn", down="kn"),
        "k_none": TilingSchedule(gate="k", down="none"),
        "k_n": TilingSchedule(gate="k", down="n"),
        "n_k": TilingSchedule(gate="n", down="k"),
        "n_kn": TilingSchedule(gate="n", down="kn"),
        "none_n": TilingSchedule(gate="none", down="n"),
        "none_none": TilingSchedule(gate="none", down="none"),
    }

    result_metrics_list = []

    tiling_schedule_to_test = list(tiling_schedule_list.keys())
    for tiling_schedule_name in tiling_schedule_to_test:
        tiling_schedule = tiling_schedule_list[tiling_schedule_name]

        # Tiling for M
        if tiling_schedule.gate in ["kn", "k", "n", "none"]:
            tile_m = B  # also tile_m_down
        else:
            tile_m = 16  # also tile_m_down

        # Tiling for K
        if tiling_schedule.gate in ["mkn", "mk", "kn", "k"]:
            tile_k = 16
        else:
            tile_k = model_config.dim

        # Tiling for N
        if tiling_schedule.gate in ["mkn", "mn", "kn", "n"]:
            tile_n = 16
        else:
            tile_n = model_config.moe_inter_dim  # also tile_k_down

        # Tiling for N_down
        if tiling_schedule.down in ["mkn", "mn", "kn", "n"]:
            tile_n_down = 16
        else:
            tile_n_down = model_config.dim

        gate_up_linear_config = LinearTileConfig(m=tile_m, k=tile_k, n=tile_n)
        down_linear_config = LinearTileConfig(m=tile_m, k=tile_n, n=tile_n_down)

        result_metrics = ResultMetrics(
            b=B,
            dim=model_config.dim,
            moe_inter_dim=model_config.moe_inter_dim,
            tiling_schedule=tiling_schedule_name,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            tile_n_down=tile_n_down,
            gate_up_compute_bw=GATE_UP_COMPUTE_BW,
            act_fn_compute_bw=ACT_FN_COMPUTE_BW,
            mult_compute_bw=MULT_COMPUTE_BW,
            down_compute_bw=DOWN_COMPUTE_BW,
            flops=sum(
                [
                    (
                        2 * B * model_config.dim * model_config.moe_inter_dim * 3
                    ),  # 3 (Linear layers)
                    B * model_config.moe_inter_dim,  # 1 (Element-wise mult)
                    (
                        8 * B * model_config.dim * model_config.moe_inter_dim
                    ),  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                ]
            ),
            simulate_mode=simulate_mode,
        )

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
            comp_bw=GATE_UP_COMPUTE_BW,
        )  # [1, B // tile_m, DIM // tile_k] (tile: [tile_m, tile_k])

        act_gate = UnaryMap(
            graph=step_graph,
            input=gate,
            fn=map_fn.Silu(),
            write_back_mu=False,
            compute_bw=ACT_FN_COMPUTE_BW,
        )  # [1, B // tile_m, DIM // tile_k] (tile: [tile_m, tile_k])

        proj = BinaryMap(step_graph, act_gate, up, map_fn.Mul(), False, MULT_COMPUTE_BW)
        # [1, B // tile_m, DIM // tile_k] (tile: [tile_m, tile_k])

        down = Linear(
            step_graph=step_graph,
            input=proj,
            weight=w_down_list[0],
            tile_config=down_linear_config,
            comp_bw=DOWN_COMPUTE_BW,
            write_back_mu=True,
            par_dispatch=par_dispatch,
        )

        output = OffChipStore(
            graph=step_graph,
            input=down,
            par_dispatch=par_dispatch,
            store_file_name="output",
        )

        step_graph = infer_broadcast(step_graph)

        # ------------ Print Graph ------------
        OUTPUT_FILENAME = (
            f"expert_{tiling_schedule_name}_{tile_m}_{tile_k}_{tile_n}_{tile_n_down}"
        )
        save_graph_format(step_graph, OUTPUT_FILENAME, ["svg", "png"])

        # ------------ Access-Reuse Analysis ------------
    #     total_off_chip_traffic = sympy.Integer(0)
    #     total_on_chip_requirement = sympy.Integer(0)

    #     for node_tuple in step_graph.nodes(data=True):
    #         node, data = node_tuple
    #         if isinstance(node, StepOps):
    #             total_off_chip_traffic = sympy.Add(
    #                 total_off_chip_traffic, node.off_chip_traffic()
    #             )
    #             total_on_chip_requirement = sympy.Add(
    #                 total_on_chip_requirement, node.on_chip_requirement()
    #             )
    #         else:
    #             raise ValueError(f"Node {node} in the graph is not a StepOps")

    #     print(f"Total on-chip requirement (bytes): {total_on_chip_requirement}")
    #     print(f"Total off-chip traffic (bytes): {total_off_chip_traffic}")

    #     result_metrics.on_chip_requirement = int(total_on_chip_requirement)  # bytes
    #     result_metrics.off_chip_traffic = int(total_off_chip_traffic)  # bytes
    #     result_metrics.operational_intensity = (
    #         result_metrics.flops / result_metrics.off_chip_traffic
    #     )  # flops/byte

    #     # ------------ Simulate ------------
    #     cycles = None
    #     if simulate_mode == "functional":
    #         n_channel = 8
    #         channel_depth = 1
    #         hbm_config = HBMConfig(64, n_channel, 2, 2, 1, 14)
    #         sim_config = SimConfig(channel_depth=channel_depth)

    #         result_metrics.mem_bw = n_channel * 32  # 32 bytes/cycle per channel
    #         result_metrics.channel_depth = channel_depth
    #         result_metrics.ridge_point = (
    #             sum(
    #                 [
    #                     result_metrics.gate_up_compute_bw,
    #                     result_metrics.act_fn_compute_bw,
    #                     result_metrics.mult_compute_bw,
    #                     result_metrics.down_compute_bw,
    #                 ]
    #             )  # FLOPs/cycle
    #             / result_metrics.mem_bw  # bytes/cycle
    #         )  # FLOPs/byte

    #         if logging is None:
    #             cycles, _, _ = simulate(
    #                 step_graph,
    #                 False,  # logging
    #                 hbm_config,
    #                 sim_config,
    #                 "./graph.pb",
    #             )
    #         else:
    #             assert isinstance(logging, str), "Logging must be a string path"
    #             cycles, _, _ = simulate(
    #                 step_graph,
    #                 True,  # logging
    #                 hbm_config,
    #                 sim_config,
    #                 "./graph.pb",
    #                 logging,
    #             )

    #         result_metrics.cycles = cycles

    #     elif simulate_mode == "timing":
    #         pass

    #     # ------------ Gold Calculation & Verification ------------

    #     if check_gold:
    #         down = gold_calc(
    #             input=input_tensor,
    #             w_gate=linear_gate_list[0],
    #             w_up=linear_up_list[0],
    #             w_down=linear_down_list[0],
    #         )
    #         print(f"Down: {output.get_untiled_shape()}")
    #         check_gold_tensor(output.store_file_name, down)

    #     result_metrics_list.append(result_metrics)

    # # ------------ Save to CSV ------------
    # if csv_filename is not None:
    #     field_names = [field.name for field in fields(ResultMetrics)]

    #     with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=field_names)

    #         # Write header
    #         writer.writeheader()

    #         # Write data rows
    #         for metrics in result_metrics_list:
    #             # Convert dataclass instance to dictionary
    #             row_data = {
    #                 field.name: getattr(metrics, field.name)
    #                 for field in fields(metrics)
    #             }
    #             writer.writerow(row_data)


def test_expert_tiling_sweep_single_schedule():
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
        "kn_k": TilingSchedule(gate="kn", down="k"),
        "kn_kn": TilingSchedule(gate="kn", down="kn"),
        "k_none": TilingSchedule(gate="k", down="none"),
        "k_n": TilingSchedule(gate="k", down="n"),
        "n_k": TilingSchedule(gate="n", down="k"),
        "n_kn": TilingSchedule(gate="n", down="kn"),
        "none_n": TilingSchedule(gate="none", down="n"),
        "none_none": TilingSchedule(gate="none", down="none"),
    }

    # ------------ Sim Conig ------------
    # simulate_mode = "full"
    simulate_mode = "timing"
    # simulate_mode = None

    check_gold = simulate_mode == "full"

    logging = False

    par_dispatch = 4

    tiling_schedule_name = "mn_mk"

    # ------------ Model Configuration ------------
    model_config = SimpleExample()

    # ------------ Batch Size ------------
    B = 64


    # ------------ Compute Bandwidths ------------
    GATE_UP_COMPUTE_BW = 4096
    ACT_FN_COMPUTE_BW = 4096
    MULT_COMPUTE_BW = 4096
    DOWN_COMPUTE_BW = 4096

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

    result_metrics_list = []

    tiling_schedule_to_test = [
        {"tile_m": 16, "tile_n": 16},
        {"tile_m": 16, "tile_n": 32},
        {"tile_m": 16, "tile_n": 64},
        {"tile_m": 16, "tile_n": 128},
        {"tile_m": 16, "tile_n": 256},
        {"tile_m": 32, "tile_n": 16},
        {"tile_m": 32, "tile_n": 32},
        {"tile_m": 32, "tile_n": 64},
        {"tile_m": 32, "tile_n": 128},
        {"tile_m": 32, "tile_n": 256},
        {"tile_m": 64, "tile_n": 16},
        {"tile_m": 64, "tile_n": 32},
        {"tile_m": 64, "tile_n": 64},
        {"tile_m": 64, "tile_n": 128},
        {"tile_m": 64, "tile_n": 256},
    ]
    for idx, tiling_size in enumerate(tiling_schedule_to_test):
        tiling_schedule = tiling_schedule_list[
            tiling_schedule_name
        ]  # TODO: change to tiling_size

        # Tiling for M
        if tiling_schedule.gate in ["kn", "k", "n", "none"]:
            tile_m = B  # also tile_m_down
        else:
            tile_m = tiling_size["tile_m"]  # also tile_m_down

        # Tiling for K
        if tiling_schedule.gate in ["mkn", "mk", "kn", "k"]:
            tile_k = tiling_size["tile_k"]
        else:
            tile_k = model_config.dim

        # Tiling for N
        if tiling_schedule.gate in ["mkn", "mn", "kn", "n"]:
            tile_n = tiling_size["tile_n"]
        else:
            tile_n = model_config.moe_inter_dim  # also tile_k_down

        # Tiling for N_down
        if tiling_schedule.down in ["mkn", "mn", "kn", "n"]:
            tile_n_down = tiling_size["tile_n_down"]
        else:
            tile_n_down = model_config.dim

        gate_up_linear_config = LinearTileConfig(m=tile_m, k=tile_k, n=tile_n)
        down_linear_config = LinearTileConfig(m=tile_m, k=tile_n, n=tile_n_down)

        result_metrics = ResultMetrics(
            b=B,
            dim=model_config.dim,
            moe_inter_dim=model_config.moe_inter_dim,
            tiling_schedule=tiling_schedule_name,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            tile_n_down=tile_n_down,
            gate_up_compute_bw=GATE_UP_COMPUTE_BW,
            act_fn_compute_bw=ACT_FN_COMPUTE_BW,
            mult_compute_bw=MULT_COMPUTE_BW,
            down_compute_bw=DOWN_COMPUTE_BW,
            flops=sum(
                [
                    (
                        2 * B * model_config.dim * model_config.moe_inter_dim * 3
                    ),  # 3 (Linear layers)
                    B * model_config.moe_inter_dim,  # 1 (Element-wise mult)
                    (
                        8 * B * model_config.dim * model_config.moe_inter_dim
                    ),  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                ]
            ),
            simulate_mode=simulate_mode,
        )

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
            comp_bw=GATE_UP_COMPUTE_BW,
        )  # [1, B // tile_m, DIM // tile_k] (tile: [tile_m, tile_k])

        act_gate = UnaryMap(
            graph=step_graph,
            input=gate,
            fn=map_fn.Silu(),
            write_back_mu=False,
            compute_bw=ACT_FN_COMPUTE_BW,
        )  # [1, B // tile_m, DIM // tile_k] (tile: [tile_m, tile_k])

        proj = BinaryMap(step_graph, act_gate, up, map_fn.Mul(), False, MULT_COMPUTE_BW)
        # [1, B // tile_m, DIM // tile_k] (tile: [tile_m, tile_k])

        down = Linear(
            step_graph=step_graph,
            input=proj,
            weight=w_down_list[0],
            tile_config=down_linear_config,
            comp_bw=DOWN_COMPUTE_BW,
            write_back_mu=True,
            par_dispatch=par_dispatch,
        )

        output = OffChipStore(
            graph=step_graph,
            input=down,
            par_dispatch=par_dispatch,
            store_file_name="output",
        )

        step_graph = infer_broadcast(step_graph)

        # ------------ Print Graph ------------
        # OUTPUT_FILENAME = (
        #     f"expert_{tiling_schedule_name}_{tile_m}_{tile_k}_{tile_n}_{tile_n_down}"
        # )
        # save_graph_format(step_graph, OUTPUT_FILENAME, ["svg"])

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

        result_metrics.on_chip_requirement = int(total_on_chip_requirement)  # bytes
        result_metrics.off_chip_traffic = int(total_off_chip_traffic)  # bytes
        result_metrics.operational_intensity = (
            result_metrics.flops / result_metrics.off_chip_traffic
        )  # flops/byte

        #     # ------------ Simulate ------------
        cycles = None
        if simulate_mode in ["full", "timing"]:
            n_channel = 8
            channel_depth = 1
            hbm_config = HBMConfig(64, n_channel, 2, 2, 1, 14)
            sim_config = SimConfig(
                channel_depth=channel_depth, functional_sim=simulate_mode == "full"
            )

            result_metrics.mem_bw = n_channel * 32  # 32 bytes/cycle per channel
            result_metrics.channel_depth = channel_depth
            result_metrics.ridge_point = (
                sum(
                    [
                        result_metrics.gate_up_compute_bw,
                        result_metrics.act_fn_compute_bw,
                        result_metrics.mult_compute_bw,
                        result_metrics.down_compute_bw,
                    ]
                )  # FLOPs/cycle
                / result_metrics.mem_bw  # bytes/cycle
            )  # FLOPs/byte

            if not logging:
                cycles, duration_ms, duration_s = simulate(
                    step_graph,
                    False,  # logging
                    hbm_config,
                    sim_config,
                    "./graph.pb",
                )
            else:
                cycles, duration_ms, duration_s = simulate(
                    step_graph,
                    True,  # logging
                    hbm_config,
                    sim_config,
                    "./graph.pb",
                    f"expert_{tiling_schedule_name}_{tile_m}_{tile_k}_{tile_n}_{tile_n_down}",
                )

            result_metrics.cycles = cycles
            result_metrics.duration_ms = duration_ms
            result_metrics.duration_s = duration_s

        # ------------ Gold Calculation & Verification ------------

        if (simulate_mode == "full") and check_gold:
            down = gold_calc(
                input=input_tensor,
                w_gate=linear_gate_list[0],
                w_up=linear_up_list[0],
                w_down=linear_down_list[0],
            )
            print(f"Down: {output.get_untiled_shape()}")
            check_gold_tensor(output.store_file_name, down)

        result_metrics_list.append(result_metrics)

    # # ------------ Save to CSV ------------
    out_file = f"./hdl_validation/fig5.csv"

    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "tile_b",
                "dim",
                "tile_inter",
                "cycles (STeP sim)",
                "off_chip_mem_traffic(MB) (STeP sim)"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics, tiling_size in zip(result_metrics_list, tiling_schedule_to_test):
                writer.writerow({
                    "tile_b": tiling_size["tile_m"],
                    "dim": model_config.dim,
                    "tile_inter": tiling_size["tile_n"],
                    "cycles (STeP sim)": metrics.cycles,
                    "off_chip_mem_traffic(MB) (STeP sim)": round(metrics.off_chip_traffic / 1e6,2),
                })
    
            print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
