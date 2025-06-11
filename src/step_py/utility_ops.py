from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import List, Tuple, Union
from step_py.datatype import MultiHot, Index, Stream
from step_py.ops import StepOps, get_stream
from networkx import MultiDiGraph


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


class SelectGen(StepOps):
    underlying: torch.Tensor
    is_multihot: bool
    _stream: Stream

    def __init__(self, is_multihot: bool, tensor: torch.Tensor):
        super().__init__()
        self.is_multihot = is_multihot
        self.underlying = tensor

        dtype = MultiHot() if is_multihot else Index()
        self._stream = Stream(dtype=dtype, shape=tuple(tensor.shape[:-1]))

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
