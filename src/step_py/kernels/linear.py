from dataclasses import dataclass
from abc import ABC


class TileConfig(ABC):
    pass


@dataclass
class TileMN(TileConfig):
    m: int
    n: int


@dataclass
class TIleMK(TileConfig):
    m: int
    k: int


@dataclass
class TileMNK(TileConfig):
    m: int
    k: int
    n: int


def Linear(input, tile_config: TileConfig):
    pass
