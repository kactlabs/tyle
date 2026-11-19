"""AMX q8_0 weight packing, spec §4.4.

Port of llama.cpp `ggml/src/ggml-cpu/amx/mmq.cpp` (`pack_B` for block_q8_0,
`pack_qs<block_q8_0>`, `s8s8_compensation`, `convert_B_packed_format`).

Weights (N, K) are quantized per 32 values along K (q8_0: int8 quants + one fp16 scale
per block). They are stored as a grid of tiles in {NB, KB} order, NB = N/16, KB = K/32.
Each tile is TILE_BYTES long:

    quants  {TILE_K/VNNI_BLK=8, TILE_N=16, VNNI_BLK=4}  int8   (512 B)  VNNI-interleaved
    d       {TILE_N}                                    fp16   (32 B)
    comp    {TILE_N}                                    int32  (64 B)  s8s8 compensation
"""

from typing import Tuple

import numpy as np

from . import constants as C
from .types import Tensor

TILE_N = 16
TILE_K = 32
VNNI_BLK = 4
QK8_0 = 32
TILE_BYTES = TILE_N * TILE_K + TILE_N * 2 + TILE_N * 4  # 608


def quantize_q8_0(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ggml `quantize_row_q8_0` reference: returns (qs int8 (N, K), d float16 (N, K//32))."""
    if x.ndim != 2 or x.shape[1] % QK8_0 != 0:
        raise ValueError(f"x must be 2-D with K a multiple of {QK8_0}, got {x.shape}")
    n, k = x.shape
    blocks = x.astype(np.float32).reshape(n, k // QK8_0, QK8_0)
    d = np.abs(blocks).max(axis=-1) / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    scaled = blocks * inv[..., None]
    qs = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).astype(np.int8)  # round half away from zero
    return qs.reshape(n, k), d.astype(np.float16)


def pack_amx_q8_0(qs: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Pack q8_0 weights into AMX tiles.

    qs: int8 (N, K); d: float16 (N, K//32). Returns uint8 (NB, KB, TILE_BYTES).
    """
    if qs.dtype != np.int8 or qs.ndim != 2:
        raise ValueError("qs must be a 2-D int8 array")
    n, k = qs.shape
    if n % TILE_N or k % TILE_K:
        raise ValueError(f"(N, K)=({n}, {k}) must be multiples of ({TILE_N}, {TILE_K})")
    if d.dtype != np.float16 or d.shape != (n, k // QK8_0):
        raise ValueError(f"d must be float16 with shape {(n, k // QK8_0)}, got {d.dtype} {d.shape}")
    nb, kb = n // TILE_N, k // TILE_K

    # (NB, 16, KB, 8 k-groups, 4) -> (NB, KB, 8 k-groups, 16 n, 4): column-major within the tile, VNNI blocks of 4
    quants = qs.reshape(nb, TILE_N, kb, TILE_K // VNNI_BLK, VNNI_BLK).transpose(0, 2, 3, 1, 4)
    quants = np.ascontiguousarray(quants).reshape(nb, kb, TILE_N * TILE_K)

    scales = np.ascontiguousarray(d.reshape(nb, TILE_N, kb).transpose(0, 2, 1))  # (NB, KB, 16)

    # comp[n] = 128 * sum of the tile's 32 quants of column n (dpbusd of 0x80 bytes against signed quants)
    comp = (128 * qs.astype(np.int32).reshape(nb, TILE_N, kb, TILE_K).sum(axis=-1)).transpose(0, 2, 1)
    comp = np.ascontiguousarray(comp.astype("<i4"))

    out = np.empty((nb, kb, TILE_BYTES), dtype=np.uint8)
    out[:, :, : TILE_N * TILE_K] = quants.view(np.uint8)
    out[:, :, TILE_N * TILE_K : TILE_N * TILE_K + TILE_N * 2] = scales.astype("<f2").view(np.uint8).reshape(nb, kb, -1)
    out[:, :, TILE_N * TILE_K + TILE_N * 2 :] = comp.view(np.uint8).reshape(nb, kb, -1)
    return out


def make_amx_q8_0_tensor(
    name: str,
    qs: np.ndarray,
    d: np.ndarray,
    exec_order_index: int = 0,
    gpu_offload_score: int = 0,
) -> Tensor:
    """Build a TYLE tensor holding AMX-packed q8_0 weights. `dims` is the logical (N, K)."""
    packed = pack_amx_q8_0(qs, d)
    return Tensor(
        name=name, dims=tuple(qs.shape), data=packed, encoding=C.ENC_AMX_INT8, quant_bits=8,
        tile_layout=C.LAYOUT_AMX_TILE, exec_order_index=exec_order_index,
        gpu_offload_score=gpu_offload_score,
        extras={
            "amx.tile_n": TILE_N,
            "amx.tile_k": TILE_K,
            "amx.vnni_blk": VNNI_BLK,
            "amx.source_quant": "q8_0",
            "amx.has_compensation": True,
        },
    )


def amx_quants() -> Tuple[str, ...]:
    """AMX source quants the native packer supports."""
    return _native().quants()


def _native():
    try:
        from . import _amxpack
    except ImportError as e:  # not built
        raise ImportError(
            "tyle._amxpack is not built; install the package (pip install -e .) to compile tyle/src/tyle_pack.cpp"
        ) from e
    return _amxpack


def pack_amx(quant: str, blocks, n: int, k: int) -> np.ndarray:
    """Pack raw ggml blocks into AMX tiles with the native packer (port of llama.cpp mmq.cpp).

    `blocks` holds N rows of K/block_k ggml blocks, exactly as in a GGUF tensor.
    Returns uint8 (N/16, K/block_k, tile_bytes), the layout llama.cpp's AMX kernels consume.
    """
    mod = _native()
    block_k, _block_bytes, tile_bytes = mod.info(quant)
    raw = mod.pack(quant, np.ascontiguousarray(blocks).tobytes() if isinstance(blocks, np.ndarray) else blocks, n, k)
    return np.frombuffer(raw, dtype=np.uint8).reshape(n // TILE_N, k // block_k, tile_bytes)


def make_amx_tensor(
    name: str,
    quant: str,
    blocks,
    n: int,
    k: int,
    exec_order_index: int = 0,
    gpu_offload_score: int = 0,
) -> Tensor:
    """Build a TYLE tensor with AMX-packed weights of any supported source quant. `dims` is (N, K)."""
    block_k, _bb, _tb = _native().info(quant)
    packed = pack_amx(quant, blocks, n, k)
    return Tensor(
        name=name, dims=(n, k), data=packed, encoding=C.ENC_AMX_INT8, quant_bits=_QUANT_BITS[quant],
        tile_layout=C.LAYOUT_AMX_TILE, exec_order_index=exec_order_index,
        gpu_offload_score=gpu_offload_score,
        extras={
            "amx.tile_n": TILE_N,
            "amx.tile_k": TILE_K,
            "amx.vnni_blk": VNNI_BLK,
            "amx.source_quant": quant,
            "amx.block_k": block_k,
            "amx.has_compensation": quant == "q8_0",
        },
    )


def amx_view(t: Tensor) -> np.ndarray:
    """Reshape a loaded AMX tensor to (NB, KB, tile_bytes), zero-copy."""
    if t.encoding != C.ENC_AMX_INT8:
        raise ValueError(f"tensor {t.name!r} is not AMX-encoded")
    quant = t.extras.get("amx.source_quant")
    block_k, _bb, tile_bytes = _native().info(quant)
    if (t.extras.get("amx.tile_n"), t.extras.get("amx.tile_k"), t.extras.get("amx.vnni_blk")) != (
        TILE_N, TILE_K, VNNI_BLK,
    ):
        raise ValueError(f"tensor {t.name!r} was packed with a different tile geometry")
    n, k = t.dims
    return np.asarray(t.data).reshape(n // TILE_N, k // block_k, tile_bytes)


_QUANT_BITS = {"q4_0": 4, "q4_1": 4, "q8_0": 8, "q4_K": 4, "q5_K": 5, "q6_K": 6, "iq4_xs": 4}


def amx_q8_0_view(t: Tensor) -> np.ndarray:
    """Reshape a loaded AMX q8_0 tensor to (NB, KB, TILE_BYTES), zero-copy."""
    if t.encoding != C.ENC_AMX_INT8 or t.extras.get("amx.source_quant") != "q8_0":
        raise ValueError(f"tensor {t.name!r} is not an AMX q8_0 tensor")
    if (t.extras.get("amx.tile_n"), t.extras.get("amx.tile_k"), t.extras.get("amx.vnni_blk")) != (
        TILE_N, TILE_K, VNNI_BLK,
    ):
        raise ValueError(f"tensor {t.name!r} was packed with a different tile geometry")
    n, k = t.dims
    return np.asarray(t.data).reshape(n // TILE_N, k // TILE_K, TILE_BYTES)
