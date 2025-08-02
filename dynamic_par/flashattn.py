from networkx import MultiDiGraph
from typing import Union, Tuple, List

from step_py.ops import *


def build_flashattn_graph(
    step_graph: MultiDiGraph,
    query: Union[StepOps, Tuple[StepOps, int]],
    key: Union[StepOps, Tuple[StepOps, int]],
    value: Union[StepOps, Tuple[StepOps, int]],
    k_cache: Union[StepOps, Tuple[StepOps, int]],
    v_cache: Union[StepOps, Tuple[StepOps, int]],
    idx_metadata: List[Union[StepOps, Tuple[StepOps, int]]],
    seq_len_metadata: List[Union[StepOps, Tuple[StepOps, int]]],
    offset_metadata: List[Union[StepOps, Tuple[StepOps, int]]],
) -> Union[StepOps, Tuple[StepOps, int]]:
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
    - query:    [DynB]       (tile: [4,D])
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
        - Used in four places (i.e., the length of the list should be 4)
            - Selection stream for the partition used to append to the last tile (2 - one each for K and V)
            - Address generation for writing back the tile with the new key and value appended to the KV cache (2 - one each for K and V)

    - offset_metadata:  [DynB] (tile: [1,1])
        - contains the offset of the incoming q,k,v
        - Used in two places (i.e., the length of the list should be 2)
            - Appending the new key and value to the last tile (2 - one each for K and V)


    Outputs:
    - output:   [DynB, DynN] (tile: [4,D])
    """

    # ------------ Stage 1: Append the new key to tiles read from the K cache ------------

    partition_k_cache = FlatPartition(
        graph=step_graph,
        input=k_cache,
        control=a,
        partition_rank=0,
        switch_cycles=[1, 1],
        write_back_mu=False,
        num_consumers=2,
    )

    # ------------ Stage 2: Wrte back the last tile with the new key and value appended to the KV cache ------------
