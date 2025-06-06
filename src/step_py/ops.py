from abc import ABC, abstractmethod, abstractproperty
import torch
from typing import List, Tuple, Union
from step_py.functions.map_fn import MapFn
from step_py.datatype import Stream, Tile, Float16, Float32
from networkx import MultiDiGraph


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
    _stream: Stream

    def __init__(
        self,
        underlying: torch.Tensor,
        stride: Tuple[int, ...],
        out_shape_tiled: Tuple[int, ...],
        tile_row: int,
        tile_col: int,
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

        if underlying.dtype == torch.float32:
            self.n_byte = 4

            stream_dtype = Tile(
                dtype=Float32(),
                shape=(tile_row, tile_col),
            )
            self._stream = Stream(dtype=stream_dtype, shape=self.out_shape_tiled)
        elif underlying.dtype == torch.float16:
            self.n_byte = 2

            stream_dtype = Tile(
                dtype=Float16(),
                shape=(tile_row, tile_col),
            )
            self._stream = Stream(dtype=stream_dtype, shape=self.out_shape_tiled)
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
        org_input: Union["StepOps", Tuple["StepOps", int]],
        new_input: Union["StepOps", Tuple["StepOps", int]],
    ):
        raise NotImplementedError(
            "Shouldn't be called for nodes that doesn't have an input stream"
        )


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
            dtype=input_stream.dtype,
            shape=tuple(input_stream.shape + (repeat_factor,)),
        )

        graph.add_edge(input, self)

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
        self.input = new_input


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
        fn: str,
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
            dtype=self.fn.apply((in1_stream.dtype, in2_stream.dtype)),
            shape=self.in1.stream.shape,
        )

        graph.add_edges_from([(in1, self), (in2, self)])

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
            Stream(dtype=in_stream.dtype, shape=in_stream.shape)
            for _ in range(num_consumers)
        ]

        graph.add_edge(input, self)

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
        self.input = new_input


class OffChipStore(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    tensor_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    store_file_name: str

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        store_file_name: str = "output.npy",
    ):
        super().__init__()

        self._input = input
        self.tensor_shape_tiled = input.stream.shape
        self.tile_row = input.stream.dtype.shape[0]
        self.tile_col = input.stream.dtype.shape[1]
        self.store_file_name = store_file_name

        graph.add_edge(input, self)

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


class FlatPartition(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]
    control: Union[StepOps, Tuple[StepOps, int]]
    num_consumers: int
    _stream: List[Stream]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        control: Union[StepOps, Tuple[StepOps, int]],
        num_consumers: int,
    ):
        super().__init__()

        self._input = input
        self.control = control
        self.num_consumers = num_consumers

        in_stream: Stream = get_stream(input)
        control_stream: Stream = get_stream(control)
        self._stream = []
        # self._stream = [
        #     Stream(dtype=in_stream.dtype, shape=TODO) for _ in range(num_consumers)
        # ]

        graph.add_edge(input, self)

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
            self.input = new_input
        elif self.control == org_input:
            if get_stream(self.control) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.control = new_input
        else:
            raise ValueError("Wrong org_input")
