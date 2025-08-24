from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from step_py.ops import *
from step_py.utility_ops import *
from step_py.functions import accum_fn, map_fn, init_fn, map_accum_fn
from proto import datatype_pb2, func_pb2, graph_pb2, ops_pb2
from step_py.datatype import *
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


@dataclass
class SimConfig:
    channel_depth: Optional[int]
    functional_sim: bool = True
    mock_bf16: bool = False
    config_dict: Optional[Dict[int, int]] = None


def simulate(
    graph: MultiDiGraph,
    logging: bool,
    hbm_config: HBMConfig,
    sim_config: SimConfig,
    protobuf_file: str,
    db_name: Optional[str] = None,
):
    serialize(graph, protobuf_file, sim_config.functional_sim)

    cycles = 0
    duration_ms = 0
    duration_s = 0

    result, cycles, duration_ms, duration_s = (
        step_perf.run_graph(  # pylint: disable=no-member
            protobuf_file, logging, hbm_config, sim_config, db_name
        )
    )
    print(f"Result: {result}")
    print(f"Cycles: {cycles}")
    print(f"Duration: {duration_ms} ms, {duration_s} s")

    return cycles, duration_ms, duration_s


# pylint: disable=no-member
def to_pb_elem_to_elem_func(
    op_fn: map_fn.MapFn,
) -> func_pb2.ElemtoElemFunc:  # pylint: disable=no-member
    func_pb = func_pb2.ElemtoElemFunc()  # pylint: disable=no-member
    if isinstance(op_fn, map_fn.Matmul):
        map_fn_pb = func_pb2.Matmul()
        map_fn_pb.weight_transposed = op_fn.weight_transposed
        func_pb.matmul.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.DynMatmul):
        map_fn_pb = func_pb2.DynMatmul()
        map_fn_pb.weight_transposed = op_fn.weight_transposed
        func_pb.dyn_matmul.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.Silu):
        map_fn_pb = func_pb2.Silu()
        func_pb.silu.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.Mul):
        map_fn_pb = func_pb2.Mul()
        func_pb.mul.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.Add):
        map_fn_pb = func_pb2.Add()
        func_pb.add.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.Div):
        map_fn_pb = func_pb2.Div()
        func_pb.div.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.RowWiseSum):
        map_fn_pb = func_pb2.RowWiseSum()
        func_pb.row_wise_sum.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.Exp):
        map_fn_pb = func_pb2.Exp()
        func_pb.exp.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.SetOffset):
        map_fn_pb = func_pb2.SetOffset()
        func_pb.set_offset.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.RowWiseAppend):
        map_fn_pb = func_pb2.RowWiseAppend()
        func_pb.row_wise_append.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_fn.CacheWriteAddrGen):
        map_fn_pb = func_pb2.CacheWriteAddrGen()
        map_fn_pb.offset_per_idx = op_fn.row_offset
        func_pb.cache_write_addr_gen.CopyFrom(map_fn_pb)
    else:
        raise NotImplementedError(
            f"Function {op_fn} is not implemented for serialization."
        )
    return func_pb


# pylint: disable=no-member
def to_pb_map_accum_func(
    op_fn: map_accum_fn.MapAccumFn,
) -> func_pb2.MapAccumFunc:  # pylint: disable=no-member
    func_pb = func_pb2.MapAccumFunc()  # pylint: disable=no-member
    if isinstance(op_fn, map_accum_fn.Matmul):
        map_fn_pb = func_pb2.Matmul()
        map_fn_pb.weight_transposed = op_fn.weight_transposed
        func_pb.matmul.CopyFrom(map_fn_pb)
    elif isinstance(op_fn, map_accum_fn.DynMatmul):
        map_fn_pb = func_pb2.DynMatmul()
        map_fn_pb.weight_transposed = op_fn.weight_transposed
        func_pb.dyn_matmul.CopyFrom(map_fn_pb)
    else:
        raise NotImplementedError(
            f"Function {op_fn} is not implemented for serialization."
        )
    return func_pb


def to_pb_accum_func(
    op_fn: accum_fn.AccumFn,
) -> func_pb2.AccumFunc:  # pylint: disable=no-member
    func_pb = func_pb2.AccumFunc()  # pylint: disable=no-member
    if isinstance(op_fn, accum_fn.Mul):
        accum_fn_pb = func_pb2.Mul()
        func_pb.mul.CopyFrom(accum_fn_pb)
    elif isinstance(op_fn, accum_fn.Add):
        accum_fn_pb = func_pb2.Add()
        func_pb.add.CopyFrom(accum_fn_pb)
    elif isinstance(op_fn, accum_fn.RetileRow):
        accum_fn_pb = func_pb2.RetileRow()
        func_pb.retile_row.CopyFrom(accum_fn_pb)
    elif isinstance(op_fn, accum_fn.SignalReqAllRead):
        accum_fn_pb = func_pb2.SignalReqAllRead()
        func_pb.signal_req_all_read.CopyFrom(accum_fn_pb)

    else:
        raise NotImplementedError(
            f"Function {op_fn} is not implemented for serialization."
        )
    return func_pb


# pylint: disable=no-member
def to_pb_init_func(op_fn: init_fn.InitFn) -> func_pb2.InitFunc:
    func_pb = func_pb2.InitFunc()
    if isinstance(op_fn, init_fn.Zero):
        func_pb.zero.CopyFrom(func_pb2.Zero())
    elif isinstance(op_fn, init_fn.Empty):
        func_pb.empty.CopyFrom(func_pb2.Empty())
    else:
        raise NotImplementedError(
            f"Function {op_fn} is not implemented for serialization."
        )

    return func_pb


def to_pb_datatype(
    dtype: Union[Tile, Buffer, Select, ElementTP],
) -> datatype_pb2.DataType:
    if isinstance(dtype, (Tile, DynTile)):
        if isinstance(dtype.tile_dtype, Float32):
            dtype_pb = datatype_pb2.DataType()
            dtype_pb.f32.CopyFrom(datatype_pb2.F32())
            return dtype_pb
        elif isinstance(dtype.tile_dtype, Float16):
            dtype_pb = datatype_pb2.DataType()
            dtype_pb.f16.CopyFrom(datatype_pb2.F16())
            return dtype_pb
        elif isinstance(dtype.tile_dtype, Uint64):
            dtype_pb = datatype_pb2.DataType()
            dtype_pb.u64.CopyFrom(datatype_pb2.U64())
            return dtype_pb
        elif isinstance(dtype.tile_dtype, Bool):
            dtype_pb = datatype_pb2.DataType()
            dtype_pb.bool.CopyFrom(datatype_pb2.Bool())
            return dtype_pb
        else:
            raise ValueError(f"Unsupported Tile datatype({dtype})")
    elif isinstance(dtype, Buffer):
        dtype_pb = datatype_pb2.DataType()
        if isinstance(dtype.buff_dtype, Tile):
            buff_pb = datatype_pb2.Buffer()

            if isinstance(dtype.buff_dtype.tile_dtype, Float32):
                buff_pb.f32.CopyFrom(datatype_pb2.F32())
            elif isinstance(dtype.buff_dtype.tile_dtype, Float16):
                buff_pb.f16.CopyFrom(datatype_pb2.F16())
            else:
                raise ValueError(
                    f"Unsupported Tile datatype({dtype.buff_dtype.tile_dtype}) for Buffer"
                )

            dtype_pb.buffer.CopyFrom(buff_pb)
            return dtype_pb
    elif isinstance(dtype, Select):
        dtype_pb = datatype_pb2.DataType()
        if isinstance(dtype, MultiHot):
            dtype_pb.multi_hot.CopyFrom(datatype_pb2.MultiHot())
            return dtype_pb
        if isinstance(dtype, Index):
            dtype_pb.index.CopyFrom(datatype_pb2.Index())
            return dtype_pb

        raise ValueError(f"Unsupported Select datatype({dtype})")
    elif isinstance(dtype, Uint64):
        dtype_pb = datatype_pb2.DataType()
        dtype_pb.scalar_u64.CopyFrom(datatype_pb2.ScalarU64())
        return dtype_pb
    elif isinstance(dtype, Bool):
        dtype_pb = datatype_pb2.DataType()
        dtype_pb.scalar_bool.CopyFrom(datatype_pb2.ScalarBool())
        return dtype_pb
    raise ValueError(f"Unsupported datatype({dtype})")


# pylint: disable=no-member
def serialize(graph: MultiDiGraph, protobuf_file: str, functional: bool):
    prog_graph = graph_pb2.ProgramGraph()  # pylint: disable=no-member
    prog_graph.name = ""

    for op_node in graph.nodes(data=True):
        operator = prog_graph.operators.add()
        if isinstance(op_node, Tuple):
            op, _ = op_node
        else:
            op = op_node
        operator.name = str(op)
        operator.id = op.instance_id

        if isinstance(op, OffChipStore):
            offchipstore_pb = ops_pb2.OffChipStore()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                offchipstore_pb.input_id = input_node.instance_id
                offchipstore_pb.stream_idx = idx
                offchipstore_pb.dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                offchipstore_pb.input_id = op.input.instance_id
                offchipstore_pb.dtype.CopyFrom(
                    to_pb_datatype(op.input.stream.stream_dtype)
                )

            offchipstore_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            offchipstore_pb.tile_row = op.tile_row
            offchipstore_pb.tile_col = op.tile_col
            if functional:
                offchipstore_pb.store_path = op.store_file_name
            offchipstore_pb.par_dispatch = op.par_dispatch

            operator.off_chip_store.CopyFrom(offchipstore_pb)
        elif isinstance(op, OffChipLoad):
            offchipload_pb = ops_pb2.OffChipLoad()
            offchipload_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            offchipload_pb.stride.extend(list(op.stride))
            offchipload_pb.out_shape_tiled.extend(list(op.out_shape_tiled))
            offchipload_pb.tile_row = op.tile_row
            offchipload_pb.tile_col = op.tile_col
            offchipload_pb.n_byte = op.n_byte
            offchipload_pb.par_dispatch = op.par_dispatch

            offchipload_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            if functional:
                file_path = f"{str(op)}.npy"
                offchipload_pb.npy_path = file_path
                np.save(file_path, op.underlying.detach().contiguous().numpy())
                print(f"Saved {str(op)} data to {file_path}")

            operator.off_chip_load.CopyFrom(offchipload_pb)
        elif isinstance(op, DynOffChipLoad):
            dyn_offchipload_pb = ops_pb2.DynOffChipLoad()

            if isinstance(op.ref, Tuple):
                ref_node, idx = op.ref
                dyn_offchipload_pb.ref_stream_idx = idx
                dyn_offchipload_pb.ref_id = ref_node.instance_id
                dyn_offchipload_pb.ref_dtype.CopyFrom(
                    to_pb_datatype(ref_node.stream_idx(idx).stream_dtype)
                )

            else:
                dyn_offchipload_pb.ref_id = op.ref.instance_id
                dyn_offchipload_pb.ref_dtype.CopyFrom(
                    to_pb_datatype(op.ref.stream.stream_dtype)
                )

            dyn_offchipload_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            dyn_offchipload_pb.stride.extend(list(op.stride))
            dyn_offchipload_pb.out_shape_tiled.extend(list(op.out_shape_tiled))
            dyn_offchipload_pb.tile_row = op.tile_row
            dyn_offchipload_pb.tile_col = op.tile_col
            dyn_offchipload_pb.n_byte = op.n_byte
            dyn_offchipload_pb.par_dispatch = op.par_dispatch

            dyn_offchipload_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            if functional:
                file_path = f"{str(op)}.npy"
                dyn_offchipload_pb.npy_path = file_path
                np.save(file_path, op.underlying.detach().contiguous().numpy())
                print(f"Saved {str(op)} data to {file_path}")

            operator.dyn_off_chip_load.CopyFrom(dyn_offchipload_pb)
        elif isinstance(op, RandomOffChipLoad):
            randomoffchipload_pb = ops_pb2.RandomOffChipLoad()

            if isinstance(op.raddr, Tuple):
                raddr_node, idx = op.raddr
                randomoffchipload_pb.raddr_stream_idx = idx
                randomoffchipload_pb.raddr_id = raddr_node.instance_id
            elif isinstance(op.raddr, StepOps):
                randomoffchipload_pb.raddr_id = op.raddr.instance_id

            randomoffchipload_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))
            randomoffchipload_pb.tile_row = op.tile_row
            randomoffchipload_pb.tile_col = op.tile_col
            randomoffchipload_pb.n_byte = op.n_byte
            randomoffchipload_pb.base_addr_byte = op.base_addr_byte
            randomoffchipload_pb.par_dispatch = op.par_dispatch

            randomoffchipload_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            if functional:
                file_path = f"{str(op)}.npy"
                randomoffchipload_pb.npy_path = file_path
                np.save(file_path, op.underlying.detach().contiguous().numpy())
                print(f"Saved {str(op)} data to {file_path}")

            operator.random_off_chip_load.CopyFrom(randomoffchipload_pb)
        elif isinstance(op, RandomOffChipStore):
            randomoffchipstore_pb = ops_pb2.RandomOffChipStore()

            if isinstance(op.waddr, Tuple):
                waddr_node, idx = op.waddr
                randomoffchipstore_pb.waddr_stream_idx = idx
                randomoffchipstore_pb.waddr_id = waddr_node.instance_id
            elif isinstance(op.waddr, StepOps):
                randomoffchipstore_pb.waddr_id = op.waddr.instance_id

            if isinstance(op.wdata, Tuple):
                wdata_node, idx = op.wdata
                randomoffchipstore_pb.wdata_stream_idx = idx
                randomoffchipstore_pb.wdata_id = wdata_node.instance_id
                randomoffchipstore_pb.wdata_dtype.CopyFrom(
                    to_pb_datatype(wdata_node.stream_idx(idx).stream_dtype)
                )
            elif isinstance(op.wdata, StepOps):
                randomoffchipstore_pb.wdata_id = op.wdata.instance_id
                randomoffchipstore_pb.wdata_dtype.CopyFrom(
                    to_pb_datatype(op.wdata.stream.stream_dtype)
                )

            randomoffchipstore_pb.tensor_shape_tiled.extend(list(op.tensor_shape_tiled))

            if functional:
                file_path = f"{str(op)}.npy"
                randomoffchipstore_pb.npy_path = file_path
                np.save(file_path, op.underlying.detach().contiguous().numpy())
                print(f"Saved {str(op)} data to {file_path}")

            randomoffchipstore_pb.tile_row = op.tile_row
            randomoffchipstore_pb.tile_col = op.tile_col
            randomoffchipstore_pb.n_byte = op.n_byte
            randomoffchipstore_pb.base_addr_byte = op.base_addr_byte
            randomoffchipstore_pb.par_dispatch = op.par_dispatch
            randomoffchipstore_pb.ack_based_on_waddr = op.ack_based_on_waddr

            operator.random_off_chip_store.CopyFrom(randomoffchipstore_pb)
        elif isinstance(op, BinaryMap):
            binarymap_pb = ops_pb2.BinaryMap()

            if isinstance(op.in1, Tuple):
                input_node, idx = op.in1
                binarymap_pb.stream_idx1 = idx
                binarymap_pb.input_id1 = input_node.instance_id
                binarymap_pb.dtype_a.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                binarymap_pb.input_id1 = op.in1.instance_id
                binarymap_pb.dtype_a.CopyFrom(
                    to_pb_datatype(op.in1.stream.stream_dtype)
                )

            if isinstance(op.in2, Tuple):
                input_node, idx = op.in2
                binarymap_pb.stream_idx2 = idx
                binarymap_pb.input_id2 = input_node.instance_id
                binarymap_pb.dtype_b.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                binarymap_pb.input_id2 = op.in2.instance_id
                binarymap_pb.dtype_b.CopyFrom(
                    to_pb_datatype(op.in2.stream.stream_dtype)
                )

            binarymap_pb.func.CopyFrom(to_pb_elem_to_elem_func(op.fn))

            binarymap_pb.compute_bw = op.compute_bw
            binarymap_pb.write_back_mu = op.write_back_mu

            binarymap_pb.dtype_out.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.binarymap.CopyFrom(binarymap_pb)
        elif isinstance(op, UnaryMap):
            binarymap_pb = ops_pb2.UnaryMap()

            if isinstance(op.input, Tuple):
                input_node, idx = op.innput
                binarymap_pb.stream_idx = idx
                binarymap_pb.input_id = input_node.instance_id
                binarymap_pb.dtype_a.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                binarymap_pb.input_id = op.input.instance_id
                binarymap_pb.dtype_a.CopyFrom(
                    to_pb_datatype(op.input.stream.stream_dtype)
                )

            binarymap_pb.func.CopyFrom(to_pb_elem_to_elem_func(op.fn))

            binarymap_pb.compute_bw = op.compute_bw
            binarymap_pb.write_back_mu = op.write_back_mu

            binarymap_pb.dtype_b.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.unarymap.CopyFrom(binarymap_pb)
        elif isinstance(op, Bufferize):
            bufferize_pb = ops_pb2.Bufferize()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                bufferize_pb.stream_idx = idx
                bufferize_pb.input_id = input_node.instance_id
            else:
                bufferize_pb.input_id = op.input.instance_id

            bufferize_pb.rank = op.rank
            bufferize_pb.dtype.CopyFrom(
                to_pb_datatype(op.stream.stream_dtype.buff_dtype)
            )

            operator.bufferize.CopyFrom(bufferize_pb)
        elif isinstance(op, Streamify):
            streamify_pb = ops_pb2.Streamify()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                streamify_pb.stream_idx = idx
                streamify_pb.input_id = input_node.instance_id
            else:
                streamify_pb.input_id = op.input.instance_id

            streamify_pb.repeat_factor.extend(list(op.repeat_factor))
            streamify_pb.rank = op.rank
            streamify_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.streamify.CopyFrom(streamify_pb)
        elif isinstance(op, DynStreamify):
            dynstreamify_pb = ops_pb2.DynStreamify()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                dynstreamify_pb.input_stream_idx = idx
                dynstreamify_pb.input_id = input_node.instance_id
            else:
                dynstreamify_pb.input_id = op.input.instance_id

            if isinstance(op.ref, Tuple):
                ref_node, idx = op.ref
                dynstreamify_pb.ref_stream_idx = idx
                dynstreamify_pb.ref_id = ref_node.instance_id
                dynstreamify_pb.ref_dtype.CopyFrom(
                    to_pb_datatype(ref_node.stream_idx(idx).stream_dtype)
                )
            else:
                dynstreamify_pb.ref_id = op.ref.instance_id
                dynstreamify_pb.ref_dtype.CopyFrom(
                    to_pb_datatype(op.ref.stream.stream_dtype)
                )

            dynstreamify_pb.bufferized_rank = op.bufferized_rank
            dynstreamify_pb.repeat_rank = op.repeat_rank
            dynstreamify_pb.input_dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.dyn_streamify.CopyFrom(dynstreamify_pb)
        elif isinstance(op, BinaryMapAccum):
            binarymapaccum_pb = ops_pb2.BinaryMapAccum()

            if isinstance(op.in1, Tuple):
                input_node, idx = op.in1
                binarymapaccum_pb.stream_idx1 = idx
                binarymapaccum_pb.input_id1 = input_node.instance_id
                binarymapaccum_pb.dtype_a.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                binarymapaccum_pb.input_id1 = op.in1.instance_id
                binarymapaccum_pb.dtype_a.CopyFrom(
                    to_pb_datatype(op.in1.stream.stream_dtype)
                )

            if isinstance(op.in2, Tuple):
                input_node, idx = op.in2
                binarymapaccum_pb.stream_idx2 = idx
                binarymapaccum_pb.input_id2 = input_node.instance_id
            else:
                binarymapaccum_pb.input_id2 = op.in2.instance_id

            binarymapaccum_pb.func.CopyFrom(to_pb_map_accum_func(op.fn))
            binarymapaccum_pb.init_func.CopyFrom(to_pb_init_func(op.init_fn))
            binarymapaccum_pb.tile_row = op.init_fn.apply().shape[0]
            binarymapaccum_pb.tile_col = op.init_fn.apply().shape[1]
            binarymapaccum_pb.rank = op.rank
            binarymapaccum_pb.compute_bw = op.compute_bw
            binarymapaccum_pb.write_back_mu = op.write_back_mu

            binarymapaccum_pb.dtype_b.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.binarymap_accum.CopyFrom(binarymapaccum_pb)
        elif isinstance(op, Accum):
            accum_pb = ops_pb2.Accum()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                accum_pb.stream_idx = idx
                accum_pb.input_id = input_node.instance_id
                accum_pb.dtype_a.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                accum_pb.input_id = op.input.instance_id
                accum_pb.dtype_a.CopyFrom(to_pb_datatype(op.input.stream.stream_dtype))

            accum_pb.func.CopyFrom(to_pb_accum_func(op.fn))
            accum_pb.init_func.CopyFrom(to_pb_init_func(op.init_fn))

            accum_row, accum_col = op.init_fn.apply().shape
            accum_pb.tile_row = accum_row
            accum_pb.tile_col = accum_col
            accum_pb.rank = op.accum_rank
            accum_pb.compute_bw = op.compute_bw
            accum_pb.write_back_mu = op.write_back_mu

            accum_pb.dtype_b.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.accum.CopyFrom(accum_pb)
        elif isinstance(op, ExpandRef):
            expandref_pb = ops_pb2.ExpandRef()

            if isinstance(op._input, Tuple):
                input_node, idx = op._input
                expandref_pb.stream_idx = idx
                expandref_pb.input_id = input_node.instance_id
                expandref_pb.dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                expandref_pb.input_id = op._input.instance_id
                expandref_pb.dtype.CopyFrom(
                    to_pb_datatype(op._input.stream.stream_dtype)
                )

            if isinstance(op.ref, Tuple):
                ref_node, idx = op.ref
                expandref_pb.ref_stream_idx = idx
                expandref_pb.ref_id = ref_node.instance_id
                expandref_pb.ref_dtype.CopyFrom(
                    to_pb_datatype(ref_node.stream_idx(idx).stream_dtype)
                )
            else:
                expandref_pb.ref_id = op.ref.instance_id
                expandref_pb.ref_dtype.CopyFrom(
                    to_pb_datatype(op.ref.stream.stream_dtype)
                )

            expandref_pb.expand_rank = op.expand_rank

            operator.expand_ref.CopyFrom(expandref_pb)

        elif isinstance(op, RepeatStatic):
            repeatstatic_pb = ops_pb2.RepeatStatic()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                repeatstatic_pb.stream_idx = idx
                repeatstatic_pb.input_id = input_node.instance_id
            else:
                repeatstatic_pb.input_id = op.input.instance_id

            repeatstatic_pb.repeat_factor = op.repeat_factor
            repeatstatic_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.repeat_static.CopyFrom(repeatstatic_pb)
        elif isinstance(op, Broadcast):
            broadcast_pb = ops_pb2.Broadcast()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                broadcast_pb.stream_idx = idx
                broadcast_pb.input_id = input_node.instance_id
            else:
                broadcast_pb.input_id = op.input.instance_id

            broadcast_pb.num_consumers = op.num_consumers
            broadcast_pb.dtype.CopyFrom(to_pb_datatype(op.stream_idx(0).stream_dtype))

            operator.broadcast.CopyFrom(broadcast_pb)
        elif isinstance(op, FlatPartition):
            flatpartition_pb = ops_pb2.FlatPartition()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                flatpartition_pb.input_stream_idx = idx
                flatpartition_pb.input_id = input_node.instance_id
                flatpartition_pb.input_dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )
            else:
                flatpartition_pb.input_id = op.input.instance_id
                flatpartition_pb.input_dtype.CopyFrom(
                    to_pb_datatype(op.input.stream.stream_dtype)
                )

            if isinstance(op.control, Tuple):
                control_node, idx = op.control
                flatpartition_pb.control_stream_idx = idx
                flatpartition_pb.control_id = control_node.instance_id
                flatpartition_pb.control_dtype.CopyFrom(
                    to_pb_datatype(control_node.stream_idx(idx).stream_dtype)
                )

            else:
                flatpartition_pb.control_id = op.control.instance_id
                flatpartition_pb.control_dtype.CopyFrom(
                    to_pb_datatype(op.control.stream.stream_dtype)
                )

            flatpartition_pb.partition_rank = op.partition_rank
            flatpartition_pb.num_consumers = op.num_consumers
            flatpartition_pb.switch_cycles.extend(list(op.switch_cycles))
            flatpartition_pb.write_back_mu = op.write_back_mu

            operator.flat_partition.CopyFrom(flatpartition_pb)
        elif isinstance(op, Parallelize):
            parallelize_pb = ops_pb2.Parallelize()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                parallelize_pb.input_stream_idx = idx
                parallelize_pb.input_id = input_node.instance_id
            else:
                parallelize_pb.input_id = op.input.instance_id

            parallelize_pb.parallelize_rank = op.parallelize_rank
            parallelize_pb.num_consumers = op.num_consumers
            parallelize_pb.switch_cycles.extend(list(op.switch_cycles))
            parallelize_pb.write_back_mu = op.write_back_mu

            parallelize_pb.input_dtype.CopyFrom(
                to_pb_datatype(op.stream_idx(0).stream_dtype)
            )

            operator.parallelize.CopyFrom(parallelize_pb)
        elif isinstance(op, FlatReassemble):
            flatreassemble_pb = ops_pb2.FlatReassemble()

            input_id_list = []
            stream_idx_list = []
            for input_stream in op._inputs:
                if isinstance(input_stream, Tuple):
                    input_node, idx = input_stream
                    input_id_list.append(input_node.instance_id)
                    stream_idx_list.append(idx)
                else:
                    input_id_list.append(input_stream.instance_id)
                    stream_idx_list.append(-1)

            flatreassemble_pb.input_id_list.extend(input_id_list)
            flatreassemble_pb.input_stream_idx_list.extend(stream_idx_list)

            if isinstance(op.control, Tuple):
                control_node, idx = op.control
                flatreassemble_pb.control_stream_idx = idx
                flatreassemble_pb.control_id = control_node.instance_id
                flatreassemble_pb.control_dtype.CopyFrom(
                    to_pb_datatype(control_node.stream_idx(idx).stream_dtype)
                )

            else:
                flatreassemble_pb.control_id = op.control.instance_id
                flatreassemble_pb.control_dtype.CopyFrom(
                    to_pb_datatype(op.control.stream.stream_dtype)
                )

            flatreassemble_pb.reassemble_rank = op.reassemble_rank
            flatreassemble_pb.switch_cycles.extend(list(op.switch_cycles))
            flatreassemble_pb.write_back_mu = op.write_back_mu
            flatreassemble_pb.input_dtype.CopyFrom(
                to_pb_datatype(op.stream.stream_dtype)
            )

            operator.flat_reassemble.CopyFrom(flatreassemble_pb)
        elif isinstance(op, EagerMerge):
            eagermerge_pb = ops_pb2.EagerMerge()

            input_id_list = []
            stream_idx_list = []
            for input_stream in op._inputs:
                if isinstance(input_stream, Tuple):
                    input_node, idx = input_stream
                    input_id_list.append(input_node.instance_id)
                    stream_idx_list.append(idx)
                else:
                    input_id_list.append(input_stream.instance_id)
                    stream_idx_list.append(-1)

            eagermerge_pb.input_id_list.extend(input_id_list)
            eagermerge_pb.input_stream_idx_list.extend(stream_idx_list)

            eagermerge_pb.input_rank = op.input_rank
            eagermerge_pb.dtype.CopyFrom(to_pb_datatype(op.stream_idx(0).stream_dtype))

            operator.eager_merge.CopyFrom(eagermerge_pb)
        elif isinstance(op, Promote):
            promote_pb = ops_pb2.Promote()

            promote_pb.input_id = op.input.instance_id
            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                promote_pb.stream_idx = idx
                promote_pb.input_id = input_node.instance_id
            else:
                promote_pb.input_id = op.input.instance_id

            promote_pb.promote_rank = op.promote_rank
            promote_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.promote.CopyFrom(promote_pb)
        elif isinstance(op, Flatten):
            flatten_pb = ops_pb2.Flatten()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                flatten_pb.stream_idx = idx
                flatten_pb.input_id = input_node.instance_id
            else:
                flatten_pb.input_id = op.input.instance_id

            flatten_pb.min_rank = op.min_rank
            flatten_pb.max_rank = op.max_rank
            flatten_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.flatten.CopyFrom(flatten_pb)
        elif isinstance(op, Reshape):
            reshape_pb = ops_pb2.Reshape()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                reshape_pb.stream_idx = idx
                reshape_pb.input_id = input_node.instance_id
            else:
                reshape_pb.input_id = op.input.instance_id

            reshape_pb.split_dim = op.reshape_rank
            reshape_pb.chunk_size = op.chunk_size
            reshape_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))
            reshape_pb.write_back_mu = op.write_back_mu
            reshape_pb.add_outer_dim = op.add_outer_dim
            reshape_pb.input_stream_rank = op.input_stream_rank

            if op.pad_fn is not None:
                reshape_pb.pad_func.CopyFrom(to_pb_init_func(op.pad_fn))
                pad_row, pad_col = op.pad_fn.apply().shape
                reshape_pb.tile_row = pad_row
                reshape_pb.tile_col = pad_col

            operator.reshape.CopyFrom(reshape_pb)
        elif isinstance(op, RetileStreamify):
            retilestreamify_pb = ops_pb2.RetileStreamify()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                retilestreamify_pb.stream_idx = idx
                retilestreamify_pb.input_id = input_node.instance_id
            else:
                retilestreamify_pb.input_id = op.input.instance_id

            retilestreamify_pb.split_row = op.split_row
            retilestreamify_pb.filter_mask = op.filter_mask
            retilestreamify_pb.dtype.CopyFrom(to_pb_datatype(op.stream.stream_dtype))

            operator.retile_streamify.CopyFrom(retilestreamify_pb)
        elif isinstance(op, SelectGen):
            selectgen_pb = ops_pb2.SelectGen()

            selectgen_pb.is_multihot = op.is_multihot

            file_path = f"{str(op)}.npy"
            selectgen_pb.npy_path = file_path
            np.save(file_path, op.underlying.detach().contiguous().numpy())
            print(f"Saved {str(op)} data to {file_path}")

            operator.select_gen.CopyFrom(selectgen_pb)
        elif isinstance(op, MetadataGen):
            metadatagen_pb = ops_pb2.MetadataGen()

            if op.underlying.dtype == torch.int64:
                dtype_pb = datatype_pb2.DataType()
                dtype_pb.scalar_i64.CopyFrom(datatype_pb2.ScalarI64())
            elif op.underlying.dtype == torch.uint64:
                dtype_pb = datatype_pb2.DataType()
                dtype_pb.scalar_u64.CopyFrom(datatype_pb2.ScalarU64())
            else:
                raise ValueError(f"Unsupported dtype: {op.underlying.dtype}")

            metadatagen_pb.dtype.CopyFrom(dtype_pb)

            file_path = f"{str(op)}.npy"
            metadatagen_pb.npy_path = file_path
            np.save(file_path, op.underlying.detach().contiguous().numpy())
            print(f"Saved {str(op)} data to {file_path}")

            operator.metadata_gen.CopyFrom(metadatagen_pb)
        elif isinstance(op, CacheReadAddrGen):
            cachereadaddrgen_pb = ops_pb2.CacheReadAddrGen()

            if isinstance(op.idx, Tuple):
                input_node, idx = op.idx
                cachereadaddrgen_pb.idx_stream_idx = idx
                cachereadaddrgen_pb.idx_id = input_node.instance_id
            else:
                cachereadaddrgen_pb.idx_id = op.idx.instance_id

            if isinstance(op.seq_len, Tuple):
                seq_len_node, idx = op.seq_len
                cachereadaddrgen_pb.seq_len_stream_idx = idx
                cachereadaddrgen_pb.seq_len_id = seq_len_node.instance_id
            else:
                cachereadaddrgen_pb.seq_len_id = op.seq_len.instance_id

            cachereadaddrgen_pb.offset_per_idx = op.row_offset

            operator.cache_read_addr_gen.CopyFrom(cachereadaddrgen_pb)
        elif isinstance(op, FilterLastTile):
            filterlasttile_pb = ops_pb2.FilterLastTile()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                filterlasttile_pb.seq_len_stream_idx = idx
                filterlasttile_pb.seq_len_id = input_node.instance_id
            else:
                filterlasttile_pb.seq_len_id = op.input.instance_id

            operator.filter_last_tile.CopyFrom(filterlasttile_pb)
        elif isinstance(op, ExpertAddrGen):
            expertaddrgen_pb = ops_pb2.ExpertAddrGen()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                expertaddrgen_pb.input_stream_idx = idx
                expertaddrgen_pb.input_id = input_node.instance_id
            else:
                expertaddrgen_pb.input_id = op.input.instance_id

            expertaddrgen_pb.num_tile_per_expert = op.num_tile_per_expert
            expertaddrgen_pb.expert_addr_base = op.expert_addr_base
            expertaddrgen_pb.dtype.CopyFrom(
                to_pb_datatype(get_stream(op.input).stream_dtype)
            )

            operator.expert_addr_gen.CopyFrom(expertaddrgen_pb)
        elif isinstance(op, PrinterContext):
            printercontext_pb = ops_pb2.PrinterContext()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                printercontext_pb.stream_idx = idx
                printercontext_pb.input_id = input_node.instance_id
                printercontext_pb.dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )

            else:
                printercontext_pb.input_id = op.input.instance_id
                printercontext_pb.dtype.CopyFrom(
                    to_pb_datatype(op.input.stream.stream_dtype)
                )

            operator.printer_context.CopyFrom(printercontext_pb)
        elif isinstance(op, PrinterContext):
            printercontext_pb = ops_pb2.PrinterContext()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                printercontext_pb.stream_idx = idx
                printercontext_pb.input_id = input_node.instance_id
                printercontext_pb.dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )

            else:
                printercontext_pb.input_id = op.input.instance_id
                printercontext_pb.dtype.CopyFrom(
                    to_pb_datatype(op.input.stream.stream_dtype)
                )

            operator.printer_context.CopyFrom(printercontext_pb)
        elif isinstance(op, ConsumerContext):
            consumercontext_pb = ops_pb2.ConsumerContext()

            if isinstance(op.input, Tuple):
                input_node, idx = op.input
                consumercontext_pb.stream_idx = idx
                consumercontext_pb.input_id = input_node.instance_id
                consumercontext_pb.dtype.CopyFrom(
                    to_pb_datatype(input_node.stream_idx(idx).stream_dtype)
                )

            else:
                consumercontext_pb.input_id = op.input.instance_id
                consumercontext_pb.dtype.CopyFrom(
                    to_pb_datatype(op.input.stream.stream_dtype)
                )

            operator.consumer_context.CopyFrom(consumercontext_pb)

        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")

    serialized_data = prog_graph.SerializeToString()

    with open(protobuf_file, "wb") as f:
        f.write(serialized_data)
    print(f"Successfully wrote to {protobuf_file}")
