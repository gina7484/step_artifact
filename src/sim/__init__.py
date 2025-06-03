from dataclasses import dataclass
from typing import List
from step_py.ops import StepOps, OffChipLoad, OffChipStore, BinaryMap, RepeatStatic
from proto import datatype_pb2, func_pb2, graph_pb2, ops_pb2
from step_py.datatype import Float32
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

    a = step_perf.run_graph(  # pylint: disable=no-member
        protobuf_file,
        logging,
        hbm_config,
    )
    print(a)


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
            offchipstore_pb.input_id = op.input.instance_id
            offchipstore_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            offchipstore_pb.tile_row = op.tile_row
            offchipstore_pb.tile_col = op.tile_col
            offchipstore_pb.store_path = op.store_file_name

            if isinstance(op.input.stream.dtype.dtype, Float32):
                dtype = datatype_pb2.DataType()
                dtype.f32.CopyFrom(datatype_pb2.F32())
                offchipstore_pb.dtype.CopyFrom(dtype)
            else:
                raise ValueError(f"Unsupported data type: {op.stream.dtype.dtype}")

            operator.off_chip_store.CopyFrom(offchipstore_pb)
        elif isinstance(op, OffChipLoad):
            offchipload_pb = ops_pb2.OffChipLoad()
            offchipload_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            offchipload_pb.stride.extend(list(op.stride))
            offchipload_pb.out_shape_tiled.extend(list(op.out_shape_tiled))
            offchipload_pb.tile_row = op.tile_row
            offchipload_pb.tile_col = op.tile_col
            offchipload_pb.n_byte = op.n_byte

            if isinstance(op.stream.dtype.dtype, Float32):
                dtype = datatype_pb2.DataType()
                dtype.f32.CopyFrom(datatype_pb2.F32())
                offchipload_pb.dtype.CopyFrom(dtype)
            else:
                raise ValueError(f"Unsupported data type: {op.stream.dtype.dtype}")

            file_path = f"{str(op)}"
            np.save(file_path, op.underlying.detach().numpy())
            offchipload_pb.npy_path = file_path + ".npy"
            print(f"Saved {str(op)} data to {file_path+ ".npy"}")

            operator.off_chip_load.CopyFrom(offchipload_pb)
        elif isinstance(op, BinaryMap):
            binarymap_pb = ops_pb2.BinaryMap()
            binarymap_pb.input_id1 = op.in1.instance_id
            binarymap_pb.input_id2 = op.in2.instance_id

            func_pb = func_pb2.ElemtoElemFunc()
            func_pb.matmul.CopyFrom(func_pb2.Matmul())
            binarymap_pb.func.CopyFrom(func_pb)

            binarymap_pb.compute_bw = op.compute_bw
            binarymap_pb.write_back_mu = op.write_back_mu

            if isinstance(op.in1.stream.dtype.dtype, Float32):
                dtype_a = datatype_pb2.DataType()
                dtype_a.f32.CopyFrom(datatype_pb2.F32())
                binarymap_pb.dtype_a.CopyFrom(dtype_a)
            else:
                raise ValueError(f"Unsupported data type: {op.stream.dtype.dtype}")

            if isinstance(op.stream.dtype.dtype, Float32):
                dtype_b = datatype_pb2.DataType()
                dtype_b.f32.CopyFrom(datatype_pb2.F32())
                binarymap_pb.dtype_b.CopyFrom(dtype_b)
            else:
                raise ValueError(f"Unsupported data type: {op.stream.dtype.dtype}")

            operator.binarymap.CopyFrom(binarymap_pb)
        elif isinstance(op, RepeatStatic):
            repeatstatic_pb = ops_pb2.RepeatStatic()
            repeatstatic_pb.input_id = op.input.instance_id
            repeatstatic_pb.repeat_factor = op.repeat_factor

            if isinstance(op.stream.dtype.dtype, Float32):
                dtype = datatype_pb2.DataType()
                dtype.f32.CopyFrom(datatype_pb2.F32())
                repeatstatic_pb.dtype.CopyFrom(dtype)
            else:
                raise ValueError(f"Unsupported data type: {op.stream.dtype.dtype}")

            operator.repeat_static.CopyFrom(repeatstatic_pb)
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")

    serialized_data = prog_graph.SerializeToString()

    with open(protobuf_file, "wb") as f:
        f.write(serialized_data)
    print(f"Successfully wrote to {protobuf_file}")
