# tyle

TYLE (Tiled Inference Layout Encoding): a CPU-native, single-file model format for local LLM inference.
Spec: https://csp.gitbook.io/genai/foss/tyle

## Plan

Progress against the spec (v0.2 draft): https://csp.gitbook.io/genai/foss/tyle

### Done

1. **Container (§1-3, §5-7):** 64-byte header with flags and magic `TYLE`, GGUF-style KV metadata,
   tensor directory, optional schedule hints and NUMA shard map, page-aligned tensor data ordered by
   `exec_order_index` when `FLAG_EXEC_ORDER` is set. The reader returns zero-copy `numpy` views over an `mmap`.
2. **LUT packing (§4.3):** `tyle.pack_lut_weights` is a port of T-MAC `preprocess_weights`.
   `tyle.make_lut_tensors` builds the `<name>` (LUT indices) and `<name>.scales` tensors with the tuning
   stored as `lut.*`. `tyle.lut_view` reshapes a loaded tensor to the kernel layout and rejects a tuning
   mismatch.
3. **AMX q8_0 packing (§4.4):** `tyle.quantize_q8_0` (ggml reference quantizer) and `tyle.pack_amx_q8_0`
   port llama.cpp `pack_B<block_q8_0>`. `tyle.make_amx_q8_0_tensor` builds the tensor (`amx.*` stored),
   `tyle.amx_q8_0_view` returns the `(N/16, K/32, 608)` tile grid. Each 608-byte tile holds 512 B of
   VNNI-interleaved int8 quants, 16 fp16 scales and 16 int32 s8s8 compensation values.

Per-tensor parameters (e.g. `lut.bits`, `amx.source_quant`) go in `Tensor.extras` and are stored in the
metadata block as `tensor.<name>.<key>`.

### Not Yet

1. **Other AMX source quants (§4.4, open item 4):** `q4_0`, `q4_1`, `q4_K`, `q5_K`, `q6_K` and `iq4_xs`
   each have their own `pack_B` layout in llama.cpp. Only `q8_0` is done.
2. **Encodings and layouts without a packer (§4.1, §4.2):** `GROUPWISE`, `CODEBOOK` and `SIMD_BLOCKED`
   exist only as enum values.
3. **Activation-sparsity block (open item 6):** `HAS_SPARSITY` is reserved in the spec but not specified.
   The reader accepts the flag and ignores any block.
4. **GPU repack (§8, open item 5):** `gpu_offload_score` is stored, but the writer never sets
   `GPU_REPACK_HINT`. The single-layout vs dual-layout decision is open.
5. **Alignment policy (open item 7):** tensor data starts on a 4096-byte page and tensors inside it are
   64-byte aligned. There is no 2 MB huge-page or straddle-avoidance logic.
6. **Versioning rules (open item 8):** the reader rejects a different major version only. Unknown
   `encoding` or `tile_layout` values are not handled.
7. **Kernels (outside the format):** nothing runs LUT or AMX matmuls, so no model output has been checked.
   The AMX packing was compared against an emulation of llama.cpp's code, not run on AMX hardware.
8. **Converter (outside the format):** nothing turns a HF or GGUF model into `.tyle` yet.
9. **Spec sync:** the LUT scales go in a separate `<name>.scales` tensor, and `lut.biased_uint` and
   `lut.scales_tensor` are extra keys. They need to be added to spec §4.3 or removed.

Not in the spec:

- The LUT scales go in a separate `<name>.scales` tensor, and `lut.biased_uint` / `lut.scales_tensor` are
  extra keys. They need to be added to spec §4.3 or removed.

## Usage

```python
import numpy as np
import tyle
from tyle import constants as C

tf = tyle.TyleFile(
    metadata={
        "tyle.arch": "llama",
        "tyle.name": "demo",
        "tyle.context_length": tyle.Typed(C.VT_U32, 2048),
        "tyle.embedding_length": tyle.Typed(C.VT_U32, 64),
        "tyle.block_count": tyle.Typed(C.VT_U32, 2),
        "tyle.head_count": tyle.Typed(C.VT_U32, 4),
        "tyle.head_count_kv": tyle.Typed(C.VT_U32, 4),
        "tyle.target_isa": ["avx512_vnni"],
        "tyle.tokenizer.model": "bpe",
        "tyle.tokenizer.tokens": ["a", "b"],
    },
    tensors=[tyle.Tensor("w", (4, 4), np.zeros((4, 4), np.float32))],
)
tyle.write(tf, "demo.tyle")
print(tyle.load("demo.tyle").tensor("w").data.shape)
```

### LUT tensors

```python
# w: uint8 (M, K), values already biased by 2**(bits-1); scales/zeros: (M, K // group_size)
params = tyle.LutParams(bits=2)  # g=4, bm=512, kfactor=16, simd_n_in=16, simd_n_out=8
tensors = tyle.make_lut_tensors("blk.0.attn_q", w, scales, zeros, params)
# ... tyle.write(tyle.TyleFile(metadata, tensors), "model.tyle")
packed = tyle.lut_view(tyle.load("model.tyle").tensor("blk.0.attn_q"), expected=params)
```

Shapes must satisfy: `K % g == 0`, `(K//g) % kfactor == 0`, `(M*bits) % bm == 0`.
The packed layout is only valid for the tuning it was built with.

### AMX q8_0 tensors

```python
qs, d = tyle.quantize_q8_0(weights)          # weights: float (N, K), N % 16 == 0, K % 32 == 0
tensor = tyle.make_amx_q8_0_tensor("blk.0.ffn_up", qs, d)
tiles = tyle.amx_q8_0_view(tyle.load("model.tyle").tensor("blk.0.ffn_up"))
```

## Install

```
/opt/miniconda3/envs/test12/bin/pip install -r requirements.txt
```
