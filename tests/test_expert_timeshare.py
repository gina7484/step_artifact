import torch
from rewrite.broadcast import infer_broadcast
from sim import HBMConfig, SimConfig, simulate
from step_py.utility_ops import *
from step_py.ops import *
from utils.draw_graph import save_graph_format
from utils.gold_checking import check_gold_tensor


def test_random_offchip_load():
    # configuration
    dim = 32
    moe_inter_dim = 128
    num_experts = 8

    tile_size = 32
    num_tile_per_expert = moe_inter_dim // tile_size

    num_selected_experts = 4

    # Expert selection (8 experts)
    routing_expert_selection = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],  # 1
            [1, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 1, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0, 0, 1],  # 7
        ]
    )  # [4,8]

    # Weights for experts
    weights = torch.randn(num_experts * dim, moe_inter_dim)

    # Step graph
    graph = MultiDiGraph()

    control = SelectGen(
        is_multihot=True,
        tensor=routing_expert_selection,
        n=8,
    )  # [1,4]

    flattened_control = Flatten(
        graph=graph,
        input=control,
        min_rank=0,
        max_rank=1,
    )  # [4]

    expert_addr_gen = ExpertAddrGen(
        graph=graph,
        input=flattened_control,
        num_tile_per_expert=num_tile_per_expert,
        expert_addr_base=0,
    )  # [4,4,1]

    random_off_chip_load = RandomOffChipLoad(
        graph=graph,
        underlying=weights,
        raddr=expert_addr_gen,
        tile_row=dim,
        tile_col=tile_size,
        base_addr_byte=0,
        par_dispatch=4,
    )
    # [4,4,1]

    # to mimic the behavior of accum
    accum_weights = Flatten(
        graph=graph,
        input=random_off_chip_load,
        min_rank=0,
        max_rank=1,
    )  # [4,4]

    reshaped_off_chip_load = Reshape(
        graph=graph,
        input=accum_weights,
        chunk_size=num_selected_experts,
        reshape_rank=1,
        write_back_mu=True,
    )  # [1,4,4]

    output = OffChipStore(
        graph=graph,
        input=reshaped_off_chip_load,
        par_dispatch=4,
        store_file_name="output",
    )

    print(output.get_untiled_shape())  # [4*32,128]

    graph = infer_broadcast(graph)

    OUTPUT_FILENAME = "test_random_offchip_load"
    save_graph_format(graph, OUTPUT_FILENAME, ["png"])

    # HBMConfig, SimConfig
    hbm_config = HBMConfig(64, 8, 2, 2, 1, 14)
    sim_config = SimConfig(channel_depth=64, functional_sim=True, mock_bf16=False)

    cycles, duration_ms, duration_s = simulate(
        graph,
        False,  # logging
        hbm_config,
        sim_config,
        "/home/ginasohn/step_tl/graph.pb",
    )

    # Gold generation
    # Define the slice ranges (corrected to get 32 elements each)
    slice_ranges = [
        slice(32, 64),  # [32:64] - indices 32 to 63 (32 elements)
        slice(0, 32),  # [0:32] - indices 0 to 31 (32 elements)
        slice(128, 160),  # [4*32:5*32] - indices 128 to 159 (32 elements)
        slice(224, 256),  # [7*32:8*32] - indices 224 to 255 (32 elements)
    ]

    # Method 1: Using list comprehension and torch.stack
    gathered_slices = [weights[slice_range, :] for slice_range in slice_ranges]
    result_tensor = torch.cat(gathered_slices, dim=0)  # [4*32, 128]

    # Verification
    check_gold_tensor(output.store_file_name, result_tensor.contiguous())
