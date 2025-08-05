from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import List, Tuple, Union
from step_py.datatype import MultiHot, Index, Stream, Tile, Uint32, Uint64
from step_py.dyndim import DynDim
from step_py.ops import StepOps, get_stream
from networkx import MultiDiGraph
import sympy


class PrinterContext(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
    ):
        super().__init__()
        self._input = input

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError("PrinterContext does not have a stream property.")

    @property
    def stream_list(self) -> List[Stream]:
        raise NotImplementedError("PrinterContext does not have a stream property.")

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

    def off_chip_traffic(self) -> int:
        """Return the off-chip traffic for this operation."""
        return 0

    def on_chip_requirement(self, count_fifos: bool = False) -> int:
        """Return the on-chip memory requirement for this operation."""
        return 0


class ConsumerContext(StepOps):
    _input: Union[StepOps, Tuple[StepOps, int]]

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
    ):
        super().__init__()
        self._input = input

        input_node = input if isinstance(input, StepOps) else input[0]
        graph.add_edge(input_node, self)

    @property
    def stream(self) -> Stream:
        raise NotImplementedError("PrinterContext does not have a stream property.")

    @property
    def stream_list(self) -> List[Stream]:
        raise NotImplementedError("PrinterContext does not have a stream property.")

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

    def off_chip_traffic(self) -> int:
        """Return the off-chip traffic for this operation."""
        return 0

    def on_chip_requirement(self, count_fifos: bool = False) -> int:
        """Return the on-chip memory requirement for this operation."""
        return 0


class FilterLastTile(StepOps):
    """
    This can be decomposed into
    - Flatmap (Counter(idX))
    - Map (IsEqual(x,y)): Generates a multihot for 0 if the condition is ture, 1 otherwise. Here, the condition is comparing whether the two elements are equal.
    """

    _input: Union[StepOps, Tuple[StepOps, int]]
    _stream: Stream

    def __init__(self, graph: MultiDiGraph, input: Union[StepOps, Tuple[StepOps, int]]):
        super().__init__()
        self._input = input
        self._stream = Stream(
            stream_dtype=MultiHot(2),
            shape=get_stream(input).shape + (DynDim(f"{str(self)}"),),
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
        if self.input == org_input:
            if get_stream(self.input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self._input = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)


class CacheReadAddrGen(StepOps):
    """
    Similar to ExpertAddrGen, this is also an coarser-grained version of collection of ops.
    This can be decomposed into
    - Map (Multly with constant): variable = idx, constant = row offset in the KV cache
    - Flatmap (Counter(idX)): the column offset for the tiles in the KV cache
    - Map (BinaryAdd): (idx * row offset) + column offset
    """

    idx: Union[StepOps, Tuple[StepOps, int]]
    seq_len: Union[StepOps, Tuple[StepOps, int]]
    row_offset: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        idx: Union[StepOps, Tuple[StepOps, int]],
        seq_len: Union[StepOps, Tuple[StepOps, int]],
        row_offset: int,
    ):
        super().__init__()
        self.idx = idx
        self.seq_len = seq_len
        self.row_offset = row_offset
        self._stream = Stream(
            stream_dtype=Tile(tile_dtype=Uint64(), shape=(1, 1)),
            shape=get_stream(idx).shape + (DynDim(f"{str(self)}"),),
        )

        idx_node = idx if isinstance(idx, StepOps) else idx[0]
        seq_len_node = seq_len if isinstance(seq_len, StepOps) else seq_len[0]
        graph.add_edge(idx_node, self)
        graph.add_edge(seq_len_node, self)

    @property
    def stream(self) -> Stream:
        """The stream of the operation."""
        return self._stream

    @property
    def stream_list(self) -> List[Stream]:
        return [self._stream]

    @property
    def input(self) -> Union["StepOps", Tuple["StepOps", int]]:
        return NotImplementedError(
            "Shouldn't be called for nodes that has multiple inputs"
        )

    @property
    def input_list(self) -> List[Union["StepOps", Tuple["StepOps", int]]]:
        return [self.idx, self.seq_len]

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
        if self.idx == org_input:
            if get_stream(self.input) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.idx = new_input
        elif self.seq_len == org_input:
            if get_stream(self.seq_len) != get_stream(new_input):
                raise ValueError("The shape of the input stream shouldn't change")
            self.seq_len = new_input
        else:
            raise ValueError("Wrong org_input")

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)


class ExpertAddrGen(StepOps):
    """
    This is expressed as a Flatmap when using the operators in the STeP abstraction.
    However, to make the simulation simpler, we make a coarser-grained version of this.
    """

    _input: Union[StepOps, Tuple[StepOps, int]]
    num_tile_per_expert: int
    expert_addr_base: int
    _stream: Stream

    def __init__(
        self,
        graph: MultiDiGraph,
        input: Union[StepOps, Tuple[StepOps, int]],
        num_tile_per_expert: int,
        expert_addr_base: int,
    ):
        super().__init__()
        self._input = input
        self.num_tile_per_expert = num_tile_per_expert
        self.expert_addr_base = expert_addr_base
        self._stream = Stream(
            stream_dtype=Tile(tile_dtype=Uint64(), shape=(1, 1)),
            shape=get_stream(input).shape + (num_tile_per_expert, 1),
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

    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        return sympy.Integer(0)

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(0)


class SelectGen(StepOps):
    underlying: torch.Tensor
    is_multihot: bool
    _stream: Stream

    def __init__(self, is_multihot: bool, tensor: torch.Tensor, n: int):
        super().__init__()
        self.is_multihot = is_multihot
        self.underlying = tensor

        dtype = MultiHot(n) if is_multihot else Index(n)
        self._stream = Stream(stream_dtype=dtype, shape=(1,) + tuple(tensor.shape[:-1]))

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

    def off_chip_traffic(self) -> int:
        """Return the off-chip traffic for this operation."""
        return 0

    def on_chip_requirement(self, count_fifos: bool = False) -> int:
        """Return the on-chip memory requirement for this operation."""
        return 0


class MetadataGen(StepOps):
    underlying: torch.Tensor
    _stream: Stream

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.underlying = tensor

        self._stream = Stream(
            stream_dtype=Tile(tile_dtype=Uint64(), shape=(1, 1)),
            shape=(1,) + tuple(tensor.shape),
        )

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

    def off_chip_traffic(self) -> int:
        """Return the off-chip traffic for this operation."""
        return 0

    def on_chip_requirement(self, count_fifos: bool = False) -> int:
        """Return the on-chip memory requirement for this operation."""
        return 0
