from abc import ABC, abstractmethod, abstractproperty
import torch
from typing import List, Tuple
from step_py.functions.map_fn import MapFn
from step_py.datatype import Stream, Tile, Float16, Float32


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
        graph: List[StepOps],
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

        graph.append(self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"


class RepeatStatic(StepOps):
    input: StepOps
    repeat_factor: int
    _stream: Stream

    def __init__(self, graph: List[StepOps], input: StepOps, repeat_factor: int):
        super().__init__()
        self.input = input
        self.repeat_factor = repeat_factor
        self._stream = Stream(
            dtype=input.stream.dtype, shape=tuple(input.stream.shape + (repeat_factor,))
        )

        graph.append(self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"


class BinaryMap(StepOps):
    in1: StepOps
    in2: StepOps
    fn: MapFn
    write_back_mu: bool  # whether the consumer is a bufferize or not
    compute_bw: int
    _stream: Stream

    def __init__(
        self,
        graph: List[StepOps],
        in1: StepOps,
        in2: StepOps,
        fn: str,
        write_back_mu: bool,
        compute_bw: int,
    ):
        assert (
            in1.stream.shape == in2.stream.shape
        ), "Input streams must have the same shape."

        super().__init__()

        self.in1 = in1
        self.in2 = in2
        self.fn = fn
        self.write_back_mu = write_back_mu
        self.compute_bw = compute_bw

        self._stream = Stream(
            dtype=self.fn.apply((self.in1.stream.dtype, self.in2.stream.dtype)),
            shape=self.in1.stream.shape,
        )

        graph.append(self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"


class OffChipStore(StepOps):
    input: StepOps
    tensor_shape_tiled: Tuple[int, ...]
    tile_row: int
    tile_col: int
    store_file_name: str

    def __init__(
        self,
        graph: List[StepOps],
        input: StepOps,
        store_file_name: str = "output.npy",
    ):
        super().__init__()

        self.input = input
        self.tensor_shape_tiled = input.stream.shape
        self.tile_row = input.stream.dtype.shape[0]
        self.tile_col = input.stream.dtype.shape[1]
        self.store_file_name = store_file_name

        graph.append(self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        raise NotImplementedError("OffChipStore does not have a stream property.")

    def get_untiled_shape(self) -> Tuple[int, ...]:
        """Get the un-tiled shape of the tensor."""
        return self.tensor_shape_tiled[:-2] + (
            self.tensor_shape_tiled[-2] * self.tile_row,
            self.tensor_shape_tiled[-1] * self.tile_col,
        )

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}_{self.instance_id}"
