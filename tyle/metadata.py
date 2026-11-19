"""Metadata KV block encoding/decoding (spec §3)."""

import struct
from typing import Any, Dict, Tuple

from . import constants as C
from .types import Typed

_SCALAR_FMT = {
    C.VT_U8: "<B",
    C.VT_I8: "<b",
    C.VT_U16: "<H",
    C.VT_I16: "<h",
    C.VT_U32: "<I",
    C.VT_I32: "<i",
    C.VT_U64: "<Q",
    C.VT_I64: "<q",
    C.VT_F32: "<f",
    C.VT_F64: "<d",
}


def _infer_type(value: Any) -> int:
    if isinstance(value, bool):
        return C.VT_BOOL
    if isinstance(value, int):
        return C.VT_I64
    if isinstance(value, float):
        return C.VT_F64
    if isinstance(value, str):
        return C.VT_STRING
    raise TypeError(f"cannot infer TYLE metadata type for {type(value).__name__}; use Typed()")


def _encode_value(vtype: int, value: Any) -> bytes:
    if vtype in _SCALAR_FMT:
        return struct.pack(_SCALAR_FMT[vtype], value)
    if vtype == C.VT_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if vtype == C.VT_STRING:
        raw = value.encode("utf-8")
        return struct.pack("<I", len(raw)) + raw
    raise ValueError(f"unsupported value type {vtype}")


def _encode_array(elem_type: int, items: list) -> bytes:
    if elem_type == C.VT_ARRAY:
        raise ValueError("nested arrays are not supported")
    out = bytearray(struct.pack("<BI", elem_type, len(items)))
    for item in items:
        out += _encode_value(elem_type, item)
    return bytes(out)


def encode_metadata(metadata: Dict[str, Any]) -> bytes:
    out = bytearray(struct.pack("<I", len(metadata)))
    for key, value in metadata.items():
        raw_key = key.encode("utf-8")
        out += struct.pack("<H", len(raw_key)) + raw_key
        if isinstance(value, Typed):
            vtype, val = value.vtype, value.value
        elif isinstance(value, (list, tuple)):
            vtype, val = C.VT_ARRAY, list(value)
        else:
            vtype, val = _infer_type(value), value
        out += struct.pack("<B", vtype)
        if vtype == C.VT_ARRAY:
            if isinstance(value, Typed):
                raise ValueError(f"{key}: Typed array needs (elem_type, items); pass a plain list instead")
            if not val:
                raise ValueError(f"{key}: empty array has no inferable element type")
            out += _encode_array(_infer_type(val[0]), val)
        else:
            out += _encode_value(vtype, val)
    return bytes(out)


def _decode_value(buf: memoryview, pos: int, vtype: int) -> Tuple[Any, int]:
    if vtype in _SCALAR_FMT:
        fmt = _SCALAR_FMT[vtype]
        (v,) = struct.unpack_from(fmt, buf, pos)
        return v, pos + struct.calcsize(fmt)
    if vtype == C.VT_BOOL:
        return bool(buf[pos]), pos + 1
    if vtype == C.VT_STRING:
        (n,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        return bytes(buf[pos : pos + n]).decode("utf-8"), pos + n
    if vtype == C.VT_ARRAY:
        elem_type, count = struct.unpack_from("<BI", buf, pos)
        pos += 5
        items = []
        for _ in range(count):
            v, pos = _decode_value(buf, pos, elem_type)
            items.append(v)
        return items, pos
    raise ValueError(f"unknown metadata value type {vtype}")


def decode_metadata(buf: memoryview, pos: int) -> Tuple[Dict[str, Any], int]:
    (count,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    out: Dict[str, Any] = {}
    for _ in range(count):
        (klen,) = struct.unpack_from("<H", buf, pos)
        pos += 2
        key = bytes(buf[pos : pos + klen]).decode("utf-8")
        pos += klen
        vtype = buf[pos]
        pos += 1
        out[key], pos = _decode_value(buf, pos, vtype)
    return out, pos
