"""TYLE file writer."""

import struct
from typing import List

from . import constants as C
from .metadata import encode_metadata
from .types import TyleFile, payload_bytes

TENSOR_ALIGN = 64


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _tensor_meta_key(name: str, key: str) -> str:
    return f"tensor.{name}.{key}"


def _encode_directory(tf: TyleFile, offsets: List[int], lengths: List[int]) -> bytes:
    entries = bytearray()
    pool = bytearray()
    for t, off, ln in zip(tf.tensors, offsets, lengths):
        raw_name = t.name.encode("utf-8")
        entries += struct.pack("<II", len(pool), len(raw_name))
        pool += raw_name
        entries += struct.pack("<B", len(t.dims))
        entries += struct.pack(f"<{len(t.dims)}Q", *t.dims)
        entries += struct.pack("<BBBB", t.encoding, t.quant_bits, t.tile_layout, 0)
        entries += struct.pack("<QQ", off, ln)
        entries += struct.pack("<II", t.exec_order_index, t.gpu_offload_score)
    return bytes(entries + pool)


def _encode_schedule(tf: TyleFile) -> bytes:
    out = bytearray(struct.pack("<I", len(tf.schedule)))
    for e in tf.schedule:
        raw = e.machine_class.encode("utf-8")
        out += struct.pack("<H", len(raw)) + raw
        out += struct.pack("<BBBBB", e.recommended_threads, e.tile_m, e.tile_n, e.tile_k, e.flags)
    return bytes(out)


def _encode_numa(tf: TyleFile) -> bytes:
    out = bytearray(struct.pack("<II", tf.numa_socket_count, len(tf.numa_shards)))
    for s in tf.numa_shards:
        if len(s.shards) != tf.numa_socket_count:
            raise ValueError(f"shard for tensor {s.tensor_index} needs {tf.numa_socket_count} entries")
        out += struct.pack("<IB", s.tensor_index, s.shard_axis)
        for off, ln in s.shards:
            out += struct.pack("<QQ", off, ln)
    return bytes(out)


def write(tf: TyleFile, path: str) -> None:
    missing = [k for k in C.REQUIRED_KEYS if k not in tf.metadata]
    if missing:
        raise ValueError(f"missing required metadata keys: {', '.join(missing)}")
    names = [t.name for t in tf.tensors]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names")

    flags = tf.flags
    if any(t.encoding == C.ENC_LUT for t in tf.tensors):
        flags |= C.FLAG_HAS_LUT
    if any(t.encoding == C.ENC_AMX_INT8 for t in tf.tensors):
        flags |= C.FLAG_HAS_AMX_TILES
    if tf.numa_shards:
        flags |= C.FLAG_NUMA_SHARDED

    # Per-tensor parameters travel in the metadata block.
    metadata = dict(tf.metadata)
    for t in tf.tensors:
        extras = dict(t.extras)
        if t.encoding == C.ENC_RAW and hasattr(t.data, "dtype"):
            extras.setdefault("dtype", t.data.dtype.str)
        for k, v in extras.items():
            metadata[_tensor_meta_key(t.name, k)] = v

    payloads = [payload_bytes(t.data) for t in tf.tensors]
    order = list(range(len(tf.tensors)))
    if flags & C.FLAG_EXEC_ORDER:
        order.sort(key=lambda i: tf.tensors[i].exec_order_index)

    offsets = [0] * len(tf.tensors)
    lengths = [len(p) for p in payloads]
    cursor = 0
    for i in order:
        cursor = _align(cursor, TENSOR_ALIGN)
        offsets[i] = cursor
        cursor += lengths[i]

    meta_blob = encode_metadata(metadata)
    meta_off = C.HEADER_SIZE
    dir_off = meta_off + len(meta_blob)
    dir_blob = _encode_directory(tf, offsets, lengths)
    sched_off = dir_off + len(dir_blob)
    sched_blob = _encode_schedule(tf) if tf.schedule else b""
    numa_off = sched_off + len(sched_blob)
    numa_blob = _encode_numa(tf) if tf.numa_shards else b""
    data_off = _align(numa_off + len(numa_blob), C.PAGE_SIZE)

    header = struct.pack(
        "<4sHHQQQQQIIQ",
        C.MAGIC,
        C.VERSION_MAJOR,
        C.VERSION_MINOR,
        meta_off,
        dir_off,
        sched_off if sched_blob else 0,
        numa_off if numa_blob else 0,
        data_off,
        len(tf.tensors),
        flags,
        0,
    )
    assert len(header) == C.HEADER_SIZE

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta_blob)
        f.write(dir_blob)
        f.write(sched_blob)
        f.write(numa_blob)
        f.write(b"\0" * (data_off - f.tell()))
        for i in order:
            f.write(b"\0" * (data_off + offsets[i] - f.tell()))
            f.write(payloads[i])
