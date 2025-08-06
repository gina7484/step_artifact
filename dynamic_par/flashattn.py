from types import NoneType
from networkx import MultiDiGraph
from typing import Dict, Union, Tuple, List
from itertools import count

from step_py.functions import accum_fn, map_accum_fn, map_fn, init_fn
from step_py.ops import *
from step_py.utility_ops import *


def build_flashattn_graph(
    par_region_idx: int,
    step_graph: MultiDiGraph,
    model_config,
    query: Union[
        StepOps, Tuple[StepOps, int]
    ],  # [DynB,1]  (tile: [query_per_kvhead,D])
    key: Union[StepOps, Tuple[StepOps, int]],  # [DynB,1]    (tile: [1,D])
    value: Union[StepOps, Tuple[StepOps, int]],  # [DynB,1]  (tile: [1,D])
    k_cache_underlying: torch.Tensor,
    v_cache_underlying: torch.Tensor,
    output_underlying: torch.Tensor,  # [B*query_per_kvhead, D]
    idx_metadata: Union[StepOps, Tuple[StepOps, int]],  # Broadcast node
    seq_len_metadata: Union[StepOps, Tuple[StepOps, int]],  # Broadcast node
    offset_metadata: Union[StepOps, Tuple[StepOps, int]],  # Broadcast node
    cache_row_offset_tiled: int,
    tile_N: int,
    cache_write_back_fifo_depth: int,
    par_dispatch: int,
    mock_bf16: bool,
    compute_bw: Dict[str, int],
    channel_dict: Dict[int, int],
) -> Tuple[Union[StepOps, Tuple[StepOps, int]], Dict[int, int], StepOps]:
    """
    Dimensions:
    - DynB: batch dimension (= dimension for requests routed to this region) (dynamic regular)
    - D: head dimension (static)
    - DynN: tiled sequence length dimension (dynamic ragged)


    Tile sizes:
    - D: tile size for the head dimension (static)
         (As it is a static dim, we use an integer for both the dimension and tile size)
    - tileN: tile size for the sequence length dimension (static)


    Inputs:
    - query:    [DynB]       (tile: [query_per_kvhead,D])
    - key:      [DynB]       (tile: [1,D])
    - value:    [DynB]       (tile: [1,D])
    - K cache:  [DynB, DynN] (tile: [tileN,D])
    - v cache:  [DynB, DynN] (tile: [tileN,D])

    - idx_metadata:     [DynB] (tile: [1,1])
        - contains the index of the incoming q,k,v
        - Used in three places (i.e., the length of the list should be 3)
            - Address generation for writing back the tile with the new key and value appended to the KV cache (2 - one each for K and V)
            - Address generation for writing back the output (1)

    - seq_len_metadata: [DynB] (tile: [1,1])
        - contains the number of tiles for each request (the tile number considers having space for appending the new key, value)
        - Used in four places (i.e., the length of the list should be query_per_kvhead)
            - Selection stream for the partition used to append to the last tile (2 - one each for K and V)
            - Address generation for writing back the tile with the new key and value appended to the KV cache (2 - one each for K and V)

    - offset_metadata:  [DynB] (tile: [1,1])
        - contains the offset of the incoming q,k,v
        - Used in two places (i.e., the length of the list should be 2)
            - Appending the new key and value to the last tile (2 - one each for K and V)


    Outputs:
    - output:   [DynB, DynN] (tile: [query_per_kvhead,D])
    """

    idx_len_counter = count(start=0)
    seq_len_counter = count(start=0)
    offset_counter = count(start=0)
    # print(next(counter))  # 0
    # print(next(counter))  # 1

    # ------------ Stage 1: Load K cache------------
    # input shape (idx): [DynB] (tile: [1,1])
    # input shape (seq_len): [DynB] (tile: [1,1])
    # output shape: [DynB, DynN] (tile: [1,1])
    k_cache_load_addr = CacheReadAddrGen(
        graph=step_graph,
        idx=(idx_metadata, next(idx_len_counter)),
        seq_len=(seq_len_metadata, next(seq_len_counter)),
        row_offset=cache_row_offset_tiled,
    )

    # input shape: [DynB, DynN] (tile: [1,1])
    # output shape: [DynB, DynN] (tile: [tileN,D])
    k_cache = RandomOffChipLoad(
        graph=step_graph,
        underlying=k_cache_underlying,
        raddr=k_cache_load_addr,
        tile_row=tile_N,
        tile_col=model_config.head_dim,
        base_addr_byte=0,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    # ------------ Stage 2: Append the new key to the last tile ------------
    control_for_partition_key = FilterLastTile(
        graph=step_graph,
        input=(seq_len_metadata, next(seq_len_counter)),
    )

    # replace the dyn dim in the control stream based on the info we already know
    control_for_partition_key.stream.shape = control_for_partition_key.stream.shape[
        :-1
    ] + (k_cache.stream.shape[-1],)

    # input shape: [DynB, DynN] (tile: [tileN,D])
    # control shape: [DynB, DynN] (dtype: multihot)
    # output shape: [DynB] (tile: [tileN,D])
    #               [DynB * (DynN - 1)] (tile: [tileN,D])
    partition_key = FlatPartition(
        graph=step_graph,
        input=k_cache,
        control=control_for_partition_key,
        partition_rank=0,
        switch_cycles=[1] * 2,
        write_back_mu=False,
        num_consumers=2,
    )

    set_offset_k = BinaryMap(
        graph=step_graph,
        in1=(partition_key, 0),
        in2=(offset_metadata, next(offset_counter)),
        fn=map_fn.SetOffset(),
        write_back_mu=False,
        compute_bw=0,
    )

    append_k = Broadcast(
        graph=step_graph,
        input=BinaryMap(
            graph=step_graph,
            in1=set_offset_k,
            in2=Flatten(
                graph=step_graph,
                input=key,
                min_rank=0,
                max_rank=1,
            ),
            fn=map_fn.RowWiseAppend(),
            write_back_mu=False,
            compute_bw=0,
        ),
        num_consumers=2,
    )

    appended_k_cache = FlatReassemble(
        graph=step_graph,
        inputs=[(append_k, 0), (partition_key, 1)],
        control=control_for_partition_key,
        reassemble_rank=0,
        switch_cycles=[1] * 2,
        write_back_mu=False,
    )

    # Replace the dyn dim in the output of reassemble as we know all the elements
    # in the control stream is one hot.
    appended_k_cache.stream.shape = appended_k_cache.stream.shape[:-1] + (1,)

    # input shape: [DynB, DynN, 1] (tile: [tileN,D])
    # output shape: [DynB, DynN] (tile: [tileN,D])
    formatted_k_cache = Flatten(
        graph=step_graph,
        input=appended_k_cache,
        min_rank=0,
        max_rank=1,
    )

    # ------------ Stage 3: Wrte back the last tile with the new key and value appended to the KV cache ------------
    k_cache_write_back_addr = BinaryMap(
        graph=step_graph,
        in1=(idx_metadata, next(idx_len_counter)),
        in2=(seq_len_metadata, next(seq_len_counter)),
        fn=map_fn.CacheWriteAddrGen(row_offset=cache_row_offset_tiled),
        write_back_mu=False,
        compute_bw=0,
    )

    k_cache_to_write_back = Broadcast(
        graph=step_graph,
        input=(append_k, 1),
        num_consumers=1,
    )

    channel_dict[k_cache_to_write_back.instance_id] = cache_write_back_fifo_depth

    k_cache_write_back = RandomOffChipStore(
        graph=step_graph,
        underlying=k_cache_underlying,
        waddr=k_cache_write_back_addr,
        wdata=(k_cache_to_write_back, 0),
        tile_row=tile_N,
        tile_col=model_config.head_dim,
        base_addr_byte=0,
        buffer_depth=cache_write_back_fifo_depth,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    _sink_k_cache_write_done = ConsumerContext(
        graph=step_graph, input=k_cache_write_back
    )

    # ------------ Stage 4: Load V cache------------
    # input shape (idx): [DynB] (tile: [1,1])
    # input shape (seq_len): [DynB] (tile: [1,1])
    # output shape: [DynB, DynN] (tile: [1,1])
    v_cache_load_addr = CacheReadAddrGen(
        graph=step_graph,
        idx=(idx_metadata, next(idx_len_counter)),
        seq_len=(seq_len_metadata, next(seq_len_counter)),
        row_offset=cache_row_offset_tiled,
    )
    v_cache_load_addr.stream.shape = k_cache_load_addr.stream.shape

    # input shape: [DynB, DynN] (tile: [1,1])
    # output shape: [DynB, DynN] (tile: [tileN,D])
    v_cache = RandomOffChipLoad(
        graph=step_graph,
        underlying=v_cache_underlying,
        raddr=v_cache_load_addr,
        tile_row=tile_N,
        tile_col=model_config.head_dim,
        base_addr_byte=0,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    # ------------ Stage 5: Append the new value to the last tile ------------
    # input shape:  [DynB,] (tile: [1,1])
    # output shape: [DynB, DynN] (dtype: multihot)
    control_for_partition_value = FilterLastTile(
        graph=step_graph,
        input=(seq_len_metadata, next(seq_len_counter)),
    )

    # replace the dyn dim in the control stream based on the info we already know
    control_for_partition_value.stream.shape = control_for_partition_value.stream.shape[
        :-1
    ] + (v_cache.stream.shape[-1],)

    # input shape: [DynB, DynN] (tile: [tileN,D])
    # control shape: [DynB, DynN] (dtype: multihot)
    # output shape: [DynB] (tile: [tileN,D])
    #               [DynB * (DynN - 1)] (tile: [DynN,D])
    partition_value = FlatPartition(
        graph=step_graph,
        input=v_cache,
        control=control_for_partition_value,
        partition_rank=0,
        switch_cycles=[1] * 2,
        write_back_mu=False,
        num_consumers=2,
    )

    set_offset_v = BinaryMap(
        graph=step_graph,
        in1=(partition_value, 0),
        in2=(offset_metadata, next(offset_counter)),
        fn=map_fn.SetOffset(),
        write_back_mu=False,
        compute_bw=0,
    )

    append_v = Broadcast(
        graph=step_graph,
        input=BinaryMap(
            graph=step_graph,
            in1=set_offset_v,
            in2=Flatten(
                graph=step_graph,
                input=value,
                min_rank=0,
                max_rank=1,
            ),
            fn=map_fn.RowWiseAppend(),
            write_back_mu=False,
            compute_bw=0,
        ),
        num_consumers=2,
    )

    appended_v_cache = FlatReassemble(
        graph=step_graph,
        inputs=[(append_v, 0), (partition_value, 1)],
        control=control_for_partition_value,
        reassemble_rank=0,
        switch_cycles=[1] * 2,
        write_back_mu=False,
    )

    # Replace the dyn dim in the output of reassemble as we know all the elements
    # in the control stream is one hot.
    appended_v_cache.stream.shape = appended_v_cache.stream.shape[:-1] + (1,)

    # input shape: [DynB, DynN, 1] (tile: [1,D])
    # output shape: [DynB, DynN] (tile: [1,D])
    formatted_v_cache = Flatten(
        graph=step_graph,
        input=appended_v_cache,
        min_rank=0,
        max_rank=1,
    )

    # ------------ Stage 6: Wrte back the last tile with the new value appended to the KV cache ------------
    v_cache_write_back_addr = BinaryMap(
        graph=step_graph,
        in1=(idx_metadata, next(idx_len_counter)),
        in2=(seq_len_metadata, next(seq_len_counter)),
        fn=map_fn.CacheWriteAddrGen(row_offset=cache_row_offset_tiled),
        write_back_mu=False,
        compute_bw=0,
    )

    v_cache_to_write_back = Broadcast(
        graph=step_graph,
        input=(append_v, 1),
        num_consumers=1,
    )

    channel_dict[v_cache_to_write_back.instance_id] = cache_write_back_fifo_depth

    v_cache_write_back = RandomOffChipStore(
        graph=step_graph,
        underlying=v_cache_underlying,
        waddr=v_cache_write_back_addr,
        wdata=(v_cache_to_write_back, 0),
        tile_row=tile_N,
        tile_col=model_config.head_dim,
        base_addr_byte=0,
        buffer_depth=cache_write_back_fifo_depth,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    _sink_v_cache_write_done = ConsumerContext(
        graph=step_graph, input=v_cache_write_back
    )

    # ------------ Stage 7: Format query ------------
    # input shape:  [DynB, 1]    (tile: [query_per_kvhead,D])
    # output shape: [DynB, DynN] (tile: [query_per_kvhead,D])
    expanded_query = ExpandRef(
        graph=step_graph,
        input=query,
        ref=formatted_k_cache,
        expand_rank=1,
    )

    # ------------ Stage 8: QKT ------------
    # input1 shape: [DynB, DynN] (tile: [query_per_kvhead,D])
    # input2 shape: [DynB, DynN] (tile: [tileN,D])
    # output shape: [DynB, DynN] (tile: [query_per_kvhead,tileN])
    qkt = BinaryMap(
        graph=step_graph,
        in1=expanded_query,
        in2=formatted_k_cache,
        fn=map_fn.Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=compute_bw["qkt"],
    )

    # ------------ Stage 9: Exp ------------
    # input shape:  [DynB, DynN] (tile: [query_per_kvhead,tileN])
    # output shape: [DynB, DynN] (tile: [query_per_kvhead,tileN])
    exp = UnaryMap(
        graph=step_graph,
        input=qkt,
        fn=map_fn.Exp(),
        write_back_mu=False,
        compute_bw=compute_bw["exp"],
    )

    # ------------ Stage 10: x V ------------
    # input1 shape: [DynB, DynN] (tile: [query_per_kvhead,tileN])
    # input2 shape: [DynB, DynN] (tile: [tileN,D])
    # output shape: [DynB, ] (tile: [query_per_kvhead,D])
    mult_v = BinaryMapAccum(
        graph=step_graph,
        in1=exp,
        in2=formatted_v_cache,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(
            shape=(model_config.query_per_kvhead, model_config.head_dim),
            dtype=Float32(),
        ),
        rank=1,
        write_back_mu=False,
        compute_bw=compute_bw["multv"],
    )

    # ------------ Stage 11: Softmax ------------
    # input shape:  [DynB, DynN] (tile: [query_per_kvhead,tileN])
    # output shape: [DynB,]      (tile: [query_per_kvhead,tileN])
    tile_dtype_exp: Tile = exp.stream.stream_dtype
    tile_wise_rowsum = Accum(
        graph=step_graph,
        input=exp,
        output_stream_dtype=Tile(
            tile_dtype=tile_dtype_exp.tile_dtype, shape=tile_dtype_exp.shape
        ),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(
            shape=tile_dtype_exp.shape, dtype=tile_dtype_exp.tile_dtype
        ),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=compute_bw["tile_wise_rowsum"],
    )

    # input shape:  [DynB,] (tile: [query_per_kvhead,tileN])
    # output shape: [DynB,] (tile: [query_per_kvhead,1])
    intra_tile_rowsum = UnaryMap(
        graph=step_graph,
        input=tile_wise_rowsum,
        fn=map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=compute_bw["intra_tile_rowsum"],
    )

    # input1 shape: [DynB,] (tile: [query_per_kvhead,D])
    # input2 shape: [DynB,] (tile: [query_per_kvhead,1])
    # output shape: [DynB,] (tile: [query_per_kvhead,D])
    softmax = BinaryMap(
        graph=step_graph,
        in1=mult_v,
        in2=intra_tile_rowsum,
        fn=map_fn.Div(),
        write_back_mu=True,
        compute_bw=compute_bw["softmax_div"],
    )

    store_output = RandomOffChipStore(
        graph=step_graph,
        underlying=output_underlying,  # [B * query_per_kvhead, D]
        waddr=(idx_metadata, next(idx_len_counter)),
        wdata=softmax,
        tile_row=model_config.query_per_kvhead,
        tile_col=model_config.head_dim,
        base_addr_byte=0,
        par_dispatch=par_dispatch,
        mock_bf16=mock_bf16,
    )

    _sink_store_done_signal = ConsumerContext(
        graph=step_graph,
        input=store_output,
    )
    store_output = NoneType
    return k_cache, channel_dict, store_output
