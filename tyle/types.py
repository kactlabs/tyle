"""Data classes for TYLE files."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Typed:
    """Metadata value with an explicit TYLE value type (e.g. Typed(VT_U32, 4096))."""

    vtype: int
    value: Any


@dataclass
class Tensor:
    """A tensor to write, or a tensor read back from a file.

    `data` holds the physical payload. For RAW tensors it is an ndarray of the
    logical shape. For LUT / AMX_INT8 tensors it is the already-packed payload
    (any ndarray or bytes); TYLE stores it untouched.
    `extras` holds per-tensor parameters (e.g. lut.bits, amx.source_quant),
    persisted in the metadata block under `tensor.<name>.<key>`.
    """

    name: str
    dims: Tuple[int, ...]
    data: Any
    encoding: int = 0
    quant_bits: int = 0
    tile_layout: int = 0
    exec_order_index: int = 0
    gpu_offload_score: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleEntry:
    machine_class: str
    recommended_threads: int
    tile_m: int
    tile_n: int
    tile_k: int
    flags: int = 0


@dataclass
class NumaShard:
    tensor_index: int
    shard_axis: int  # 0=row, 1=col
    shards: List[Tuple[int, int]]  # per socket: (offset from tensor_data_offset, length)


@dataclass
class TyleFile:
    metadata: Dict[str, Any]
    tensors: List[Tensor]
    flags: int = 0
    schedule: List[ScheduleEntry] = field(default_factory=list)
    numa_socket_count: int = 0
    numa_shards: List[NumaShard] = field(default_factory=list)
    version: Tuple[int, int] = (0, 1)

    def tensor(self, name: str) -> Optional[Tensor]:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


def payload_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    return np.ascontiguousarray(data).tobytes()
