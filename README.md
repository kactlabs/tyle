# tyle

TYLE (Tiled Inference Layout Encoding): a CPU-native, single-file model format for local LLM inference.
Spec: https://csp.gitbook.io/genai/foss/tile

## Status

Basic container implemented per spec v0.2 draft:

- 64-byte header with flags, magic `TYLE`, little-endian
- GGUF-style KV metadata block (all value types, arrays of scalars/strings)
- Tensor directory with encoding, quant bits, tile layout, exec order, GPU offload score
- Optional schedule hints block and NUMA shard map
- Page-aligned tensor data, ordered by `exec_order_index` when `FLAG_EXEC_ORDER` is set
- Reader returns zero-copy `numpy` views over an `mmap`

- LUT packing (spec §4.3): `tyle.pack_lut_weights` is a port of T-MAC `preprocess_weights`.
  `tyle.make_lut_tensors` builds the `<name>` (LUT indices) and `<name>.scales` tensors with the
  tuning parameters stored as `lut.*`. `tyle.lut_view` reshapes a loaded tensor to the kernel layout
  and rejects a tuning mismatch.

Not implemented yet: kernels, AMX packing (payloads are stored as given), sparsity block.
Per-tensor parameters (e.g. `lut.bits`, `amx.source_quant`) go in `Tensor.extras` and are stored in the
metadata block as `tensor.<name>.<key>`.

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

## Install

```
/opt/miniconda3/envs/test12/bin/pip install -r requirements.txt
```
