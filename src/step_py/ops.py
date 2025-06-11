from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import List, Tuple, Union
from step_py.functions.map_fn import MapFn
from step_py.datatype import Buffer, Stream, Tile, Select, Float16, Float32
from networkx import MultiDiGraph
import sympy


def get_stream(input: Union["StepOps", Tuple["StepOps", int]]) -> Stream:
    if isinstance(input, StepOps):
        return input.stream

    if (
        isinstance(input, Tuple)
        and isinstance(input[0], StepOps)
        and isinstance(input[1], int)
    ):
        input_node, stream_idx = input
        return input_node.stream_idx(stream_idx)
    else:
        raise TypeError("Wrong input type!")


class StepOps(ABC):
    _counter: int = 0
    instance_id: int

    def __init__(self):
        self.instance_id = StepOps._counter
        StepOps._counter += 1

    @property
    @abstractmethod
    def stream(self) -> Stream:
        """The stream of the operation."""
        pass

    @property
    @abstractmethod
    def stream_list(self) -> List[Stream]:
        pass

    @property
    @abstractmethod
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        pass

    @property
    @abstractmethod
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        pass

    @abstractmethod
    def stream_idx(self, idx: int) -> Stream:
        """The stream of the operation."""
        pass

    @abstractmethod
    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        pass


class OffChipLoad(StepOps):
    underlying: torch.Tensor
    tensor_shape_tiled: Tuple[int, ...]
    stride: Tuple[int, ...]
    out_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    n_byte: int
    par_dispatch: int
    _stream: Stream

    def __init__(
        self,
        underlying: torch.Tensor,
        stride: Tuple[int, ...],
        out_shape_tiled: Tuple[int, ...],
        tile_row: int,
        tile_col: int,
        par_dispatch: int,
    ):
        super().__init__()

        self.underlying = underlying
        self.tensor_shape_tiled = tuple(
            list(underlying.shape[:-2])
            + [
                underlying.shape[-2] // tile_row,
                underlying.shape[-1] // tile_col,
            ]
        )
        self.stride = stride
        self.out_shape_tiled = out_shape_tiled
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.par_dispatch = par_dispatch

        if underlying.dtype == torch.float32:
            self.n_byte = 4

            stream_dtype = Tile(
                tile_dtype=Float32(),
                shape=(tile_row, tile_col),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.out_shape_tiled
            )
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                tile_dtype=Float16(),
                shape=(tile_row, tile_col),
            )
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=(1,) + self.out_shape_tiled
            )
        else:
            raise ValueError(f"Unsupported dtype: {underlying.dtype}")

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union[StepOps, Tuple[StepOps, int]],
        new_input: Union[StepOps, Tuple[StepOps, int]],
    ):
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )


class DynOffChipLoad(StepOps):
    ref: Union[StepOps, Tuple[StepOps, int]]
    underlying: torch.Tensor
    tensor_shape_tiled: Tuple[int, ...]
    stride: Tuple[int, ...]
    out_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    n_byte: int
    par_dispatch: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        ref: Union[StepOps, Tuple[StepOps, int]],
        underlying: torch.Tensor,
        stride: Tuple[int, ...],
        out_shape_tiled: Tuple[int, ...],
        tile_row: int,
        tile_col: int,
        par_dispatch: int,
    ):
        super().__init__()

        self.ref = ref
        self.underlying = underlying
        self.tensor_shape_tiled = tuple(
            list(underlying.shape[:-2])
            + [
                underlying.shape[-2] // tile_row,
                underlying.shape[-1] // tile_col,
            ]
        )
        self.stride = stride
        self.out_shape_tiled = out_shape_tiled
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.par_dispatch = par_dispatch

        ref_node = ref if isinstance(ref, StepOps) else ref[0]
        graph.add_edge(ref_node, self)

        if underlying.dtype == torch.float32:
            self.n_byte = 4

            stream_dtype = Tile(
                tile_dtype=Float32(),
                shape=(tile_row, tile_col),
            )
            ref_stream: Stream = get_stream(ref)
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=ref_stream.shape + self.out_shape_tiled
            )
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                tile_dtype=Float16(),
                shape=(tile_row, tile_col),
            )
            ref_stream: Stream = get_stream(ref)
            self._stream = Stream(
                stream_dtype=stream_dtype, shape=ref_stream.shape + self.out_shape_tiled
            )
        else:
            raise ValueError(f"Unsupported dtype: {underlying.dtype}")

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.ref

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.ref]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.ref) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self.ref = new_input


class RepeatStatic(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    repeat_factor: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        repeat_factor: int,
    ):
        super().__init__()
        self._input = input
        self.repeat_factor = repeat_factor

        input_stream: Stream = get_stream(input)
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=tuple(input_stream.shape + (repeat_factor,)),
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class Promote(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    promote_rank: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        promote_rank: int,
    ):
        super().__init__()
        self._input = input
        self.promote_rank = promote_rank

        input_stream: Stream = get_stream(input)
        stream_shape = list(input_stream.shape)
        stream_shape.insert(len(input_stream.shape) - promote_rank, 1)
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=tuple(stream_shape),
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class BinaryMap(StepOps):
    in1: Union[StepOps, Tuple[StepOps, int]]
    in2: Union[StepOps, Tuple[StepOps, int]]
    fn: MapFn
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        in1: Union[StepOps, Tuple[StepOps, int]],
        in2: Union[StepOps, Tuple[StepOps, int]],
        fn: MapFn,
        write_back_mu: bool,
        compute_bw: int,
    ):

        super().__init__()

        self.in1 = in1
        self.in2 = in2
        self.fn = fn
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in1_stream: Stream = get_stream(in1)
        in2_stream: Stream = get_stream(in2)

        assert (
            in1_stream.shape == in2_stream.shape
        ), "Input streams must have the same shape."

        self._stream = Stream(
            stream_dtype=self.fn.apply(
                (in1_stream.stream_dtype, in2_stream.stream_dtype)
            ),
            shape=self.in1.stream.shape,
        )

        input_node1 = in1 if isinstance(in1, StepOps) else in1[0]
        input_node2 = in2 if isinstance(in2, StepOps) else in2[0]

        graph.add_edges_from([(input_node1, self), (input_node2, self)])

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.in1

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.in1, self.in2]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.in1 == org_input:
            if get_stream(self.in1) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in1 = new_input
        elif self.in2 == org_input:
            if get_stream(self.in2) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in2 = new_input
        else:
            raise ValueError("Wrong org_input")


class BinaryMapAccum(StepOps):
    in1: Union[StepOps, Tuple[StepOps, int]]
    in2: Union[StepOps, Tuple[StepOps, int]]
    fn: MapFn
    accum_tile_row: int
    accum_tile_col: int
    rank: int
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    # [Genghan] We need an init function?
    def __init__(
        self,
        graph: MultiDiGraph,
        in1: Union[StepOps, Tuple[StepOps, int]],
        in2: Union[StepOps, Tuple[StepOps, int]],
        fn: MapFn,
        rank: int,
        write_back_mu: bool,
        compute_bw: int,
    ):

        super().__init__()

        self.in1 = in1
        self.in2 = in2
        self.fn = fn
        self.rank = rank
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in1_stream: Stream = get_stream(in1)
        in2_stream: Stream = get_stream(in2)

        assert rank > 0, "Rank must be greater than 0."
        assert (
            in1_stream.shape == in2_stream.shape
        ), f"Input streams must have the same shape. {in1_stream.shape} != {in2_stream.shape}"

        self._stream = Stream(
            stream_dtype=self.fn.apply(
                (in1_stream.stream_dtype, in2_stream.stream_dtype)
            ),
            shape=self.in1.stream.shape[: -self.rank],
        )

        self.accum_tile_row = self._stream.stream_dtype.shape[-2]
        self.accum_tile_col = self._stream.stream_dtype.shape[-1]

        input_node1 = in1 if isinstance(in1, StepOps) else in1[0]
        input_node2 = in2 if isinstance(in2, StepOps) else in2[0]

        graph.add_edges_from([(input_node1, self), (input_node2, self)])

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self.in1

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.in1, self.in2]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.in1 == org_input:
            if get_stream(self.in1) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in1 = new_input
        elif self.in2 == org_input:
            if get_stream(self.in2) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.in2 = new_input
        else:
            raise ValueError("Wrong org_input")


class Broadcast(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    num_consumers: int
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        num_consumers: int,
    ):
        super().__init__()

        self._input = input
        self.num_consumers = num_consumers

        in_stream: Stream = get_stream(input)
        self._stream = [
            Stream(stream_dtype=in_stream.stream_dtype, shape=in_stream.shape)
            for _ in range(num_consumers)
        ]

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError(
            "This property shouldn't be used for nodes with multiple output streams"
        )

    @property
    def stream_list(self) -> List[Stream]:
        return self._stream

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        return self._stream[idx]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class OffChipStore(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    tensor_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    store_file_name: str  # This should not include the file extension!!
    par_dispatch: int

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        par_dispatch: int,
        store_file_name: str = "output",
    ):
        super().__init__()

        self._input = input
        self.tensor_shape_tiled = input.stream.shape[1:]
        self.tile_row = input.stream.stream_dtype.shape[0]
        self.tile_col = input.stream.stream_dtype.shape[1]
        self.store_file_name = store_file_name
        self.par_dispatch = par_dispatch

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError("OffChipStore does not have a stream property.")

    @property
    def stream_list(self) -> List[Stream]:
        raise NotImplementedError("OffChipStore does not have a stream property.")

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes without an output stream"
        )

    def get_untiled_shape(self) -> Tuple[int, ...]:
        """Get the un-tiled shape of the tensor."""
        if len(self.tensor_shape_tiled) == 1:
            return (self.tensor_shape_tiled[-1] * self.tile_row,
                    self.tile_col)
        else:
            return self.tensor_shape_tiled[:-2] + (
                self.tensor_shape_tiled[-2] * self.tile_row,
                self.tensor_shape_tiled[-1] * self.tile_col,
            )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class Bufferize(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    rank: int
    _stream: Stream

    def __init__(
        self, graph: MultiDiGraph, input: Union[StepOps, Tuple[StepOps, int]], rank: int
    ):
        super().__init__()

        self._input = input
        self.rank = rank

        in_stream: Stream = get_stream(input)
        assert rank > 0, "Rank must be greater than 0."
        assert isinstance(
            in_stream.stream_dtype, Tile
        ), "Input stream must be a Tile type."

        buffer_shape = tuple(in_stream.shape[-rank:])
        self._stream = Stream(
            stream_dtype=Buffer(buff_dtype=in_stream.stream_dtype, shape=buffer_shape),
            shape=in_stream.shape[: -self.rank],
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class Streamify(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    repeat_factor: List[int]
    rank: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        repeat_factor: List[int],
        rank: int,  # The rank of the Buffer
    ):
        super().__init__()

        self._input = input
        self.repeat_factor = repeat_factor
        self.rank = rank

        in_stream: Stream = get_stream(input)
        assert rank > 0, "Rank must be greater than 0."
        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Input stream must be a Buffer type."

        buffer_shape = in_stream.stream_dtype.shape
        buffer_dtype = in_stream.stream_dtype.buff_dtype
        self._stream = Stream(
            stream_dtype=buffer_dtype,
            shape=in_stream.shape + tuple(repeat_factor) + buffer_shape,
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class DynStreamify(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    ref: Union[StepOps, Tuple[StepOps, int]]
    repeat_rank: int
    bufferized_rank: int
    _stream: Stream

    # [Genghan] There is an ExpandRef hidden in the operation
    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        ref: Union[StepOps, Tuple[StepOps, int]],
        repeat_rank: int,  # Starting from this rank to rank 0, the input_stream should have 1s
        bufferized_rank: int,
    ):
        super().__init__()

        self._input = input
        self.ref = ref
        self.repeat_rank = repeat_rank
        self.bufferized_rank = bufferized_rank

        in_stream: Stream = get_stream(input)
        ref_stream: Stream = get_stream(ref)

        assert bufferized_rank > 0, "Bufferized rank must be greater than 0."
        calc_rank = self.repeat_rank + 1
        assert (
            in_stream.shape[:-calc_rank] == ref_stream.shape[:-calc_rank]
        ), f"Shapes up to the repeat rank don't match: {in_stream.shape[: -calc_rank]} != {ref_stream.shape[: -calc_rank]}"

        assert (
            in_stream.shape[-calc_rank:] == (1,) * calc_rank
        ), f"Input stream shape must have 1s in the repeat rank dimensions {in_stream.shape[-calc_rank :]} != {(1,) * calc_rank}."

        assert isinstance(
            in_stream.stream_dtype, Buffer
        ), "Input stream must be a Buffer type."

        buffer_shape = in_stream.stream_dtype.shape
        buffer_dtype = in_stream.stream_dtype.buff_dtype

        self._stream = Stream(
            stream_dtype=buffer_dtype, shape=ref_stream.shape + buffer_shape
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

        ref_node = ref if isinstance(ref, StepOps) else ref[0]
        graph.add_edge(ref_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.ref]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self._input == org_input:
            if get_stream(self._input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.ref == org_input:
            if get_stream(self.ref) != get_stream(new_input):
                raise ValueError("The shape of the ref stream shouldn't change")
            self.ref = new_input
        else:
            raise ValueError("Wrong org_input")


class FlatPartition(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    control: Union[StepOps, Tuple[StepOps, int]]
    num_consumers: int
    partition_rank: int
    switch_cycles: List[int]
    write_back_mu: bool
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        control: Union[StepOps, Tuple[StepOps, int]],
        partition_rank: int,
        switch_cycles: List[int],
        write_back_mu: bool,
        num_consumers: int,
    ):
        super().__init__()

        self._input = input
        self.control = control
        self.num_consumers = num_consumers
        self.partition_rank = partition_rank
        self.switch_cycles = switch_cycles
        self.write_back_mu = write_back_mu

        input_node = input if isinstance(input, StepOps) else input[0]
        control_node = control if isinstance(control, StepOps) else control[0]
        graph.add_edge(input_node, self)
        graph.add_edge(control_node, self)

        in_stream: Stream = get_stream(input)
        # [Genghan] A trick: StepOps should use the same control_node to align the outermost dimension
        new_names = sympy.symbols(f"{str(control_node)}_0:{num_consumers}")
        self._stream = [
            Stream(
                stream_dtype=in_stream.stream_dtype,
                shape=(new_names[i],)
                + in_stream.shape[len(in_stream.shape) - partition_rank :],
            )
            for i in range(num_consumers)
        ]

    @property
    def stream(self) -> Stream:
        raise NotImplementedError(
            "This property shouldn't be used for nodes with multiple output streams"
        )

    @property
    def stream_list(self) -> List[Stream]:
        return self._stream

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input, self.control]

    def stream_idx(self, idx: int) -> Stream:
        return self._stream[idx]

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if self.input == org_input:
            if get_stream(self.input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        elif self.control == org_input:
            if get_stream(self.control) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.control = new_input
        else:
            raise ValueError("Wrong org_input")


class FlatReassemble(StepOps):
    _inputs: List[Union[StepOps, Tuple[StepOps, int]]]
    control: Union[StepOps, Tuple[StepOps, int]]
    reassemble_rank: int
    switch_cycles: List[int]
    write_back_mu: bool
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        inputs: List[Union[StepOps, Tuple[StepOps, int]]],
        control: Union[StepOps, Tuple[StepOps, int]],
        reassemble_rank: int,  # Remove dimensions at rank larger or equal to this value
        switch_cycles: List[int],
        write_back_mu: bool,
    ):
        super().__init__()

        self._inputs = inputs
        self.control = control
        self.reassemble_rank = reassemble_rank
        self.switch_cycles = switch_cycles
        self.write_back_mu = write_back_mu

        in_streams = [get_stream(input) for input in inputs]
        assert all(
            stream.shape[len(stream.shape) - reassemble_rank :]
            == in_streams[0].shape[len(in_streams[0].shape) - reassemble_rank :]
            for stream in in_streams
        ), "All input streams must have the same shape for the last 'reassemble_rank' dimensions."
        control_stream: Stream = get_stream(control)
        new_name = sympy.Symbol(f"{str(self)}_dyn")
        self._stream = Stream(
            stream_dtype=in_streams[0].stream_dtype,
            shape=control_stream.shape
            + (new_name,)
            + in_streams[0].shape[len(in_streams[0].shape) - reassemble_rank :],
        )

        for input_node in inputs:
            node = input_node if isinstance(input_node, StepOps) else input_node[0]
            graph.add_edge(node, self)

        control_node = control if isinstance(control, StepOps) else control[0]
        graph.add_edge(control_node, self)

    @property
    def stream(self) -> Stream:
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        raise NotImplementedError(
            "Shouldn't be called for nodes that has multiple input streams"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return self._inputs + [self.control]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        for i, input_node in enumerate(self._inputs):
            if input_node == org_input:
                if get_stream(input_node) != get_stream(new_input):
                    raise ValueError("The shape of the input stream shouldn't change")
                self._inputs[i] = new_input
                return

        if self.control == org_input:
            if get_stream(self.control) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.control = new_input
        else:
            raise ValueError("Wrong org_input")


class UnaryMap(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    fn: MapFn
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        fn: MapFn,
        write_back_mu: bool,
        compute_bw: int,
    ):
        super().__init__()

        self._input = input
        self.fn = fn
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in_stream: Stream = get_stream(input)

        self._stream = Stream(
            stream_dtype=self.fn.apply((in_stream.stream_dtype,)),
            shape=in_stream.shape,
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self._input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input

class Accum(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    fn: MapFn
    accum_rank: int
    write_back_mu: bool
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        output_stream_dtype: Union[Tile, Buffer, Select],
        fn: MapFn,
        accum_rank: int,
        write_back_mu: bool,
        compute_bw: int,
    ):
        super().__init__()

        self._input = input
        self.fn = fn
        self.accum_rank = accum_rank
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        in_stream: Stream = get_stream(input)
        assert accum_rank > 0, "Accum rank must be greater than 0."
        self._stream = Stream(
            stream_dtype=self.fn.apply((in_stream.stream_dtype, output_stream_dtype)),
            shape=in_stream.shape[: -self.accum_rank],
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self._input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input


class Flatten(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    min_rank: int
    max_rank: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        min_rank: int,
        max_rank: int
    ):
        super().__init__()
        self._input = input
        self.min_rank = min_rank
        self.max_rank = max_rank

        input_stream: Stream = get_stream(input)
        self._stream = Stream(
            stream_dtype=input_stream.stream_dtype,
            shape=tuple(
                self._compute_flattened_shape(input_stream.shape, min_rank, max_rank)
            ),
        )

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    def _compute_flattened_shape(self, shape, min_rank, max_rank):
        # Convert ranks to indices (rank 0 = rightmost = highest index)
        min_index = len(shape) - 1 - max_rank  # Note: max_rank gives min_index
        max_index = len(shape) - 1 - min_rank  # Note: min_rank gives max_index
        
        # Validate indices
        if min_index < 0 or max_index >= len(shape) or min_index > max_index:
            raise ValueError("Invalid rank range")
        
        # Calculate merged dimension
        merged_dim = 1
        for i in range(min_index, max_index + 1):
            merged_dim *= shape[i]
        
        # Build new shape
        new_shape = shape[:min_index] + (merged_dim,) + shape[max_index + 1:]
        
        return new_shape

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return self._input

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self._input]

    def stream_idx(self, idx: int) -> Stream:
        raise NotImplementedError(
            "Shouldn't be called for nodes that only have a single output stream"
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"

    def replace_input(
        self,
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        if get_stream(self.input) != get_stream(new_input):
            raise ValueError("The shape of the input stream shouldn't change")
        self._input = new_input
