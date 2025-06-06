from dataclasses import dataclass
from typing import List, Tuple
from step_py.ops import *
from proto import datatype_pb2, func_pb2, graph_pb2, ops_pb2
from step_py.datatype import ElementTP, Float32
import numpy as np
import step_perf


@dataclass
class HBMConfig:
    addr_offset: int
    channel_num: int
    per_channel_latency: int
    per_channel_init_interval: int
    per_channel_outstanding: int
    per_channel_start_up_time: int


def simulate(graph: List[StepOps], logging: bool, hbm_config: HBMConfig):
    protobuf_file = "/home/ginasohn/step_tl/graph.pb"

    serialize(graph, protobuf_file)

    # a = step_perf.run_graph(  # pylint: disable=no-member
    #     protobuf_file,
    #     logging,
    #     hbm_config,
    # )
    # print(a)


# pylint: disable=no-member
def to_pb_datatype(dtype: ElementTP) -> datatype_pb2.DataType:
    if isinstance(dtype, Float32):
        dtype_pb = datatype_pb2.DataType()
        dtype_pb.f32.CopyFrom(datatype_pb2.F32())
        return dtype_pb

    raise ValueError(f"Unsupported datatype({dtype}) for serialization")


# pylint: disable=no-member
def serialize(graph: List[StepOps], protobuf_file: str):
    prog_graph = graph_pb2.ProgramGraph()  # pylint: disable=no-member
    prog_graph.name = ""

    for op in graph:
        operator = prog_graph.operators.add()
        operator.name = str(op)
        operator.id = op.instance_id

        if isinstance(op, OffChipStore):
            offchipstore_pb = ops_pb2.OffChipStore()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                offchipstore_pb.input_id = input_node.instance_id
                offchipstore_pb.stream_idx = idx
                offchipstore_pb.dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).dtype.dtype)
                )
            else:
                offchipstore_pb.input_id = op.input.instance_id
                offchipstore_pb.dtype.CopyFrom(
                    to_pb_datatype(op.input.stream.dtype.dtype)
                )

            offchipstore_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            offchipstore_pb.tile_row = op.tile_row
            offchipstore_pb.tile_col = op.tile_col
            offchipstore_pb.store_path = op.store_file_name

            operator.off_chip_store.CopyFrom(offchipstore_pb)
        elif isinstance(op, OffChipLoad):
            offchipload_pb = ops_pb2.OffChipLoad()
            offchipload_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            offchipload_pb.stride.extend(list(op.stride))
            offchipload_pb.out_shape_tiled.extend(list(op.out_shape_tiled))
            offchipload_pb.tile_row = op.tile_row
            offchipload_pb.tile_col = op.tile_col
            offchipload_pb.n_byte = op.n_byte

            offchipload_pb.dtype.CopyFrom(to_pb_datatype(op.stream.dtype.dtype))

            file_path = f"{str(op)}"
            np.save(file_path, op.underlying.detach().numpy())
            offchipload_pb.npy_path = file_path + ".npy"
            print(f"Saved {str(op)} data to {file_path+ ".npy"}")

            operator.off_chip_load.CopyFrom(offchipload_pb)
        elif isinstance(op, BinaryMap):
            binarymap_pb = ops_pb2.BinaryMap()

            if isinstance(op.in1, Tuple):
                input_node, idx = op.in1
                binarymap_pb.stream_idx1 = idx
                binarymap_pb.input_id1 = input_node.instance_id
                binarymap_pb.dtype_a.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).dtype.dtype)
                )
            else:
                binarymap_pb.input_id1 = op.in1.instance_id
                binarymap_pb.dtype_a.CopyFrom(to_pb_datatype(op.in1.stream.dtype.dtype))

            if isinstance(op.in2, Tuple):
                input_node, idx = op.in2
                binarymap_pb.stream_idx2 = idx
                binarymap_pb.input_id2 = input_node.instance_id
            else:
                binarymap_pb.input_id2 = op.in2.instance_id

            func_pb = func_pb2.ElemtoElemFunc()
            func_pb.matmul.CopyFrom(func_pb2.Matmul())
            binarymap_pb.func.CopyFrom(func_pb)

            binarymap_pb.compute_bw = op.compute_bw
            binarymap_pb.write_back_mu = op.write_back_mu

            binarymap_pb.dtype_b.CopyFrom(to_pb_datatype(op.stream.dtype.dtype))

            operator.binarymap.CopyFrom(binarymap_pb)
        elif isinstance(op, RepeatStatic):
            repeatstatic_pb = ops_pb2.RepeatStatic()

            repeatstatic_pb.input_id = op.input.instance_id
            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                repeatstatic_pb.stream_idx = idx
                repeatstatic_pb.input_id = input_node.instance_id
            else:
                repeatstatic_pb.input_id = op.input.instance_id

            repeatstatic_pb.repeat_factor = op.repeat_factor
            repeatstatic_pb.dtype.CopyFrom(to_pb_datatype(op.stream.dtype.dtype))

            operator.repeat_static.CopyFrom(repeatstatic_pb)
        elif isinstance(op, Broadcast):
            broadcast_pb = ops_pb2.Broadcast()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                broadcast_pb.stream_idx = idx
                broadcast_pb.input_id = input_node.instance_id
            else:
                broadcast_pb.input_id = op.input.instance_id

            broadcast_pb.dtype.CopyFrom(to_pb_datatype(op.stream_idx(0).dtype.dtype))

            operator.broadcast.CopyFrom(broadcast_pb)
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")

    serialized_data = prog_graph.SerializeToString()

    with open(protobuf_file, "wb") as f:
        f.write(serialized_data)
    print(f"Successfully wrote to {protobuf_file}")
