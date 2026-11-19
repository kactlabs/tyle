"""Constants for the TYLE format (spec v0.1/v0.2 draft). Little-endian throughout."""

MAGIC = b"TYLE"
VERSION_MAJOR = 0
VERSION_MINOR = 1
HEADER_SIZE = 64
PAGE_SIZE = 4096

# Header flags (§2.1)
FLAG_EXEC_ORDER = 1 << 0
FLAG_HAS_LUT = 1 << 1
FLAG_HAS_AMX_TILES = 1 << 2
FLAG_HAS_SPARSITY = 1 << 3
FLAG_NUMA_SHARDED = 1 << 4
FLAG_GPU_REPACK_HINT = 1 << 5

# Metadata value types (§3)
VT_U8, VT_I8, VT_U16, VT_I16, VT_U32, VT_I32 = 0, 1, 2, 3, 4, 5
VT_U64, VT_I64, VT_F32, VT_F64, VT_BOOL, VT_STRING, VT_ARRAY = 6, 7, 8, 9, 10, 11, 12

# Tensor encoding (§4.1)
ENC_RAW, ENC_GROUPWISE, ENC_LUT, ENC_CODEBOOK, ENC_AMX_INT8 = 0, 1, 2, 3, 4

# Tile layout (§4.2)
LAYOUT_LINEAR, LAYOUT_SIMD_BLOCKED, LAYOUT_AMX_TILE, LAYOUT_LUT_INDEXED = 0, 1, 2, 3

# Required metadata keys (§3)
REQUIRED_KEYS = (
    "tyle.arch",
    "tyle.name",
    "tyle.context_length",
    "tyle.embedding_length",
    "tyle.block_count",
    "tyle.head_count",
    "tyle.head_count_kv",
    "tyle.target_isa",
    "tyle.tokenizer.model",
    "tyle.tokenizer.tokens",
)
