"""Byte-exact check of tyle's AMX packers against llama.cpp's own pack_B code.

1. Cuts the packing code out of tyle/src/mmq_ref.cpp verbatim (no retyping).
2. Compiles it with tyle/src/verify_llamacpp.cpp against SIMDe, which implements the
   AVX2/AVX-512 intrinsics on any CPU (so this works on arm64 too).
3. Packs the same random ggml blocks with llama.cpp's code and with tyle._amxpack and
   compares the bytes.

Usage (from the repo root, after `pip install -e .`):
    python tools/verify_llamacpp.py
Env: SIMDE_DIR (path to a SIMDe checkout; cloned into build/simde if unset), CXX (default clang++).
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tyle" / "src"
BUILD = ROOT / "build"
SIMDE_URL = "https://github.com/simd-everywhere/simde.git"


def between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def extract_pack_code() -> Path:
    """Write the verbatim llama.cpp fragments the harness needs to build/llamacpp_pack_extract.inc."""
    text = (SRC / "mmq_ref.cpp").read_text()
    parts = [
        between(text, "// type traits", "template <typename T>\nstruct do_unpack"),
        between(text, "// we need an extra 16 * 4B", "template <typename TB, int BLOCK_K>\nint get_row_size"),
        between(text, "// transpose utils", "void quantize_row_q8_K_vnni"),
        between(text, "#define MM256_SET_M128I", "template<typename TB, typename packed_B_t = packed_B_type<TB>>"),
    ]
    marker = "#define PACKED_INDEX"
    i = text.index(marker)
    parts.append(text[i : text.index("\n", i) + 1])
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "llamacpp_pack_extract.inc"
    out.write_text("// generated from tyle/src/mmq_ref.cpp by tools/verify_llamacpp.py\n" + "\n".join(parts))
    return out


def simde_dir() -> Path:
    env = os.environ.get("SIMDE_DIR")
    if env:
        return Path(env)
    path = BUILD / "simde"
    if not path.exists():
        subprocess.run(["git", "clone", "--depth", "1", SIMDE_URL, str(path)], check=True)
    return path


def build_reference() -> ctypes.CDLL:
    extract_pack_code()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = BUILD / f"libllamacpp_ref{suffix}"
    cmd = [
        os.environ.get("CXX", "clang++"), "-std=c++17", "-O1", "-shared", "-fPIC", "-w",
        "-DSIMDE_ENABLE_NATIVE_ALIASES",
        # portable C fallbacks: llama.cpp's q5_K code shifts by a loop variable, which x86 allows but
        # SIMDe's NEON path rejects (it needs a compile-time constant)
        "-DSIMDE_NO_NATIVE",
        f"-I{simde_dir()}", f"-I{SRC}", f"-I{BUILD}",
        str(SRC / "verify_llamacpp.cpp"), "-o", str(lib),
    ]
    subprocess.run(cmd, check=True)
    ref = ctypes.CDLL(str(lib))
    ref.ref_tile_size_of.argtypes = [ctypes.c_char_p]
    ref.ref_pack.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    return ref


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from tyle import _amxpack

    ref = build_reference()
    rng = np.random.default_rng(12345)
    failures = 0
    for quant in _amxpack.quants():
        block_k, block_bytes, tile_bytes = _amxpack.info(quant)
        ref_tile = ref.ref_tile_size_of(quant.encode())
        if ref_tile != tile_bytes:
            print(f"FAIL {quant}: tile size tyle={tile_bytes} llama.cpp={ref_tile}")
            failures += 1
            continue
        for n, kb in [(16, 1), (32, 2), (48, 3)]:
            k = block_k * kb
            raw = rng.integers(0, 256, size=n * kb * block_bytes, dtype=np.uint8).tobytes()
            out = ctypes.create_string_buffer((n // 16) * kb * tile_bytes)
            ref.ref_pack(quant.encode(), raw, n, k, out)
            want = out.raw
            got = _amxpack.pack(quant, raw, n, k)
            if got == want:
                print(f"ok   {quant:7s} N={n:3d} K={k:4d} tile={tile_bytes}B  {len(got)} bytes identical")
            else:
                bad = np.flatnonzero(np.frombuffer(got, np.uint8) != np.frombuffer(want, np.uint8))
                print(f"FAIL {quant:7s} N={n} K={k}: {bad.size} bytes differ, first at {bad[:8].tolist()}")
                failures += 1
    print("ALL MATCH" if not failures else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
