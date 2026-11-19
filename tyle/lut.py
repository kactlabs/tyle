"""LUT (T-MAC style) weight packing, spec §4.3.

`pack_lut_weights` is a port of T-MAC's `python/t_mac/weights.py::preprocess_weights`
(https://github.com/microsoft/T-MAC). The packed layout is valid only for the tuning
parameters it was built with (bm, kfactor, simd_n_in, simd_n_out), so those are stored
with the tensor and `lut_view` refuses to reinterpret a payload on a mismatch.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import constants as C
from .types import Tensor


@dataclass(frozen=True)
class LutParams:
    bits: int = 4
    g: int = 4
    bm: int = 512
    kfactor: int = 16
    simd_n_in: int = 16
    simd_n_out: int = 8

    def validate(self, m_in: int, k: int) -> None:
        """Check that (m_in, k) weights can be packed with these parameters."""
        if self.bits not in (1, 2, 3, 4):
            raise ValueError(f"bits must be 1-4, got {self.bits}")
        if self.g not in (1, 2, 4):
            raise ValueError(f"g must divide 8 (1, 2 or 4), got {self.g}")
        m_total = m_in * self.bits
        mgroup = (8 // self.g) * self.simd_n_in
        checks = [
            (k % self.g == 0, f"K={k} must be a multiple of g={self.g}"),
            ((k // self.g) % self.kfactor == 0, f"K//g={k // self.g} must be a multiple of kfactor={self.kfactor}"),
            (m_total % self.bm == 0, f"M*bits={m_total} must be a multiple of bm={self.bm}"),
            (self.bm % mgroup == 0, f"bm={self.bm} must be a multiple of (8//g)*simd_n_in={mgroup}"),
            (m_in % self.simd_n_out == 0, f"M={m_in} must be a multiple of simd_n_out={self.simd_n_out}"),
            (self.bm % self.bits == 0, f"bm={self.bm} must be a multiple of bits={self.bits}"),
            ((self.bm // self.bits) % self.simd_n_out == 0,
             f"bm//bits={self.bm // self.bits} must be a multiple of simd_n_out={self.simd_n_out}"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    def packed_shape(self, m_in: int, k: int) -> Tuple[int, int, int]:
        """Shape of the packed index array: (M*bits//bm, K//g, bm//(8//g))."""
        self.validate(m_in, k)
        return (m_in * self.bits // self.bm, k // self.g, self.bm // (8 // self.g))


def pack_lut_weights(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    params: LutParams = LutParams(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack biased-uint weights (M, K) and their scales/zeros for the LUT kernel.

    `w` is uint8 with each value already biased by 2**(bits-1) (int2 -2..1 -> 0..3).
    Returns (packed weights, permuted scales[+zeros]), as T-MAC's preprocess_weights does.
    """
    if w.dtype != np.uint8 or w.ndim != 2:
        raise ValueError("w must be a 2-D uint8 array")
    bits, g, bm, kfactor = params.bits, params.g, params.bm, params.kfactor
    simd_n_in, simd_n_out = params.simd_n_in, params.simd_n_out
    m_in, k = w.shape
    params.validate(m_in, k)
    if int(w.max()) >= (1 << bits):
        raise ValueError(f"weight values must be < 2**bits ({1 << bits}); apply the 2**(bits-1) bias first")
    if zeros is not None and zeros.shape != scales.shape:
        raise ValueError("zeros must have the same shape as scales")

    m = m_in * bits
    ngroups_per_elem = 8 // g

    # (M, K, bits) -> (M, bits, K) -> (M, bits, K//g, g)
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    w = w.transpose(0, 2, 1).reshape(m // bits, bits, k // g, g)
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    w = w.reshape(m // bits // simd_n_out, simd_n_out, bits, k // g).transpose(0, 2, 1, 3)
    mgroup = ngroups_per_elem * simd_n_in
    w = w.reshape(m // mgroup, ngroups_per_elem, simd_n_in, k // g).transpose(0, 2, 1, 3)
    w = w.reshape(m // bm, bm // mgroup, simd_n_in, ngroups_per_elem, k // g // kfactor, kfactor)
    w = w.transpose(0, 4, 1, 5, 2, 3)
    w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
    w = w.reshape(m // bm, k // g // kfactor, bm // mgroup, kfactor, simd_n_in)
    w = np.ascontiguousarray(w.reshape(m // bm, k // g, bm // ngroups_per_elem)).astype(np.uint8)

    if scales.size >= m // bits:
        if scales.ndim != 2 or scales.shape[0] != m_in or k % scales.shape[1] != 0:
            raise ValueError(f"per-group scales must have shape (M={m_in}, K//group_size), got {scales.shape}")
        group_size = k // scales.shape[1]
        scales = scales.reshape(m // bm, bm // bits, k // group_size).transpose(0, 2, 1)
        scales = scales.reshape(m // bm, k // group_size, bm // bits // simd_n_out, simd_n_out)
        if zeros is not None:
            zeros = zeros.reshape(m // bm, bm // bits, k // group_size).transpose(0, 2, 1)
            zeros = zeros.reshape(m // bm, k // group_size, bm // bits // simd_n_out, simd_n_out)
            scales = np.stack([scales, zeros], axis=-2)
        scales = scales.reshape(m // bm, k // group_size, -1)
    elif zeros is not None:
        scales = np.concatenate([scales, zeros])
    return w, np.ascontiguousarray(scales)


def make_lut_tensors(
    name: str,
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    params: LutParams = LutParams(),
    exec_order_index: int = 0,
    gpu_offload_score: int = 0,
) -> List[Tensor]:
    """Pack weights and return two TYLE tensors: `<name>` (LUT indices) and `<name>.scales`.

    `dims` of the weight tensor is the logical (M, K); the packed shape is rebuilt from
    the stored lut.* parameters by `lut_view`.
    """
    packed, packed_scales = pack_lut_weights(w, scales, zeros, params)
    m_in, k = w.shape
    per_group = scales.size >= m_in
    extras = {
        "lut.bits": params.bits,
        "lut.g": params.g,
        "lut.bm": params.bm,
        "lut.kfactor": params.kfactor,
        "lut.simd_n_in": params.simd_n_in,
        "lut.simd_n_out": params.simd_n_out,
        "lut.group_size": (k // scales.shape[1]) if per_group else k,
        "lut.has_zeros": zeros is not None,
        "lut.biased_uint": True,
        "lut.scales_tensor": f"{name}.scales",
    }
    weight = Tensor(
        name=name, dims=(m_in, k), data=packed, encoding=C.ENC_LUT, quant_bits=params.bits,
        tile_layout=C.LAYOUT_LUT_INDEXED, exec_order_index=exec_order_index,
        gpu_offload_score=gpu_offload_score, extras=extras,
    )
    scale_tensor = Tensor(
        name=f"{name}.scales", dims=packed_scales.shape, data=packed_scales, encoding=C.ENC_RAW,
        exec_order_index=exec_order_index, gpu_offload_score=gpu_offload_score,
    )
    return [weight, scale_tensor]


def lut_params(t: Tensor) -> LutParams:
    """Read the stored LUT tuning parameters of a loaded tensor."""
    if t.encoding != C.ENC_LUT:
        raise ValueError(f"tensor {t.name!r} is not LUT-encoded")
    try:
        return LutParams(
            bits=t.extras["lut.bits"], g=t.extras["lut.g"], bm=t.extras["lut.bm"],
            kfactor=t.extras["lut.kfactor"], simd_n_in=t.extras["lut.simd_n_in"],
            simd_n_out=t.extras["lut.simd_n_out"],
        )
    except KeyError as e:
        raise ValueError(f"tensor {t.name!r} is missing LUT parameter {e}") from None


def lut_view(t: Tensor, expected: Optional[LutParams] = None) -> np.ndarray:
    """Reshape a loaded LUT tensor payload to the kernel layout (M*bits//bm, K//g, bm//(8//g)), zero-copy.

    If `expected` is given (the tuning the runtime kernel was compiled for), a mismatch
    with the stored parameters raises, per spec §4.3.
    """
    stored = lut_params(t)
    if expected is not None and expected != stored:
        raise ValueError(f"LUT tuning mismatch for {t.name!r}: file has {stored}, kernel expects {expected}")
    m_in, k = t.dims
    return np.asarray(t.data).reshape(stored.packed_shape(m_in, k))
