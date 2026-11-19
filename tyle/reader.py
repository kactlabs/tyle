"""TYLE file reader. Tensor payloads are zero-copy views over an mmap."""

import mmap
import struct
from typing import Dict, List, Tuple

import numpy as np

from . import constants as C
from .metadata import decode_metadata
from .types import NumaShard, ScheduleEntry, Tensor, TyleFile


class TyleFormatError(ValueError):
    pass


def _read_directory(buf: memoryview, pos: int, count: int) -> Tuple[List[dict], int]:
    raw = []
    for _ in range(count):
        name_off, name_len, n_dims = struct.unpack_from("<IIB", buf, pos)
        pos += 9
        dims = struct.unpack_from(f"<{n_dims}Q", buf, pos)
        pos += 8 * n_dims
        enc, bits, layout, _res = struct.unpack_from("<BBBB", buf, pos)
        pos += 4
        data_off, data_len = struct.unpack_from("<QQ", buf, pos)
        pos += 16
        order, gpu = struct.unpack_from("<II", buf, pos)
        pos += 8
        raw.append(
            dict(name_off=name_off, name_len=name_len, dims=dims, encoding=enc, quant_bits=bits,
                 tile_layout=layout, data_off=data_off, data_len=data_len, order=order, gpu=gpu)
        )
    return raw, pos  # `pos` is where the name pool starts


def _read_schedule(buf: memoryview, pos: int) -> List[ScheduleEntry]:
    (n,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    out = []
    for _ in range(n):
        (mlen,) = struct.unpack_from("<H", buf, pos)
        pos += 2
        mc = bytes(buf[pos : pos + mlen]).decode("utf-8")
        pos += mlen
        threads, tm, tn, tk, fl = struct.unpack_from("<BBBBB", buf, pos)
        pos += 5
        out.append(ScheduleEntry(mc, threads, tm, tn, tk, fl))
    return out


def _read_numa(buf: memoryview, pos: int) -> Tuple[int, List[NumaShard]]:
    sockets, n = struct.unpack_from("<II", buf, pos)
    pos += 8
    shards = []
    for _ in range(n):
        idx, axis = struct.unpack_from("<IB", buf, pos)
        pos += 5
        per_socket = []
        for _ in range(sockets):
            per_socket.append(struct.unpack_from("<QQ", buf, pos))
            pos += 16
        shards.append(NumaShard(idx, axis, per_socket))
    return sockets, shards


def load(path: str) -> TyleFile:
    """Parse a .tyle file. Tensor `data` are read-only views backed by an mmap."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    buf = memoryview(mm)
    if len(buf) < C.HEADER_SIZE or bytes(buf[:4]) != C.MAGIC:
        raise TyleFormatError("not a TYLE file (bad magic)")

    (major, minor, meta_off, dir_off, sched_off, numa_off, data_off, count, flags, reserved) = (
        struct.unpack_from("<HHQQQQQIIQ", buf, 4)
    )
    if major != C.VERSION_MAJOR:
        raise TyleFormatError(f"unsupported major version {major}")
    if reserved != 0 or flags >> 6:
        raise TyleFormatError("reserved header bits are set")

    metadata, _ = decode_metadata(buf, meta_off)
    raw, pool_pos = _read_directory(buf, dir_off, count)

    tensors: List[Tensor] = []
    for r in raw:
        name = bytes(buf[pool_pos + r["name_off"] : pool_pos + r["name_off"] + r["name_len"]]).decode("utf-8")
        start = data_off + r["data_off"]
        end = start + r["data_len"]
        if end > len(buf):
            raise TyleFormatError(f"tensor {name!r} extends past end of file")
        prefix = f"tensor.{name}."
        extras: Dict[str, object] = {k[len(prefix):]: v for k, v in metadata.items() if k.startswith(prefix)}
        payload = np.frombuffer(buf[start:end], dtype=np.uint8)
        dtype = extras.get("dtype")
        if r["encoding"] == C.ENC_RAW and isinstance(dtype, str):
            payload = payload.view(np.dtype(dtype)).reshape(r["dims"])
        tensors.append(
            Tensor(name=name, dims=tuple(r["dims"]), data=payload, encoding=r["encoding"],
                   quant_bits=r["quant_bits"], tile_layout=r["tile_layout"],
                   exec_order_index=r["order"], gpu_offload_score=r["gpu"], extras=extras)
        )

    # Keep tensor-scoped keys out of the model-level metadata view.
    model_meta = {k: v for k, v in metadata.items() if not k.startswith("tensor.")}
    schedule = _read_schedule(buf, sched_off) if sched_off else []
    sockets, shards = _read_numa(buf, numa_off) if numa_off else (0, [])
    return TyleFile(metadata=model_meta, tensors=tensors, flags=flags, schedule=schedule,
                    numa_socket_count=sockets, numa_shards=shards, version=(major, minor))
