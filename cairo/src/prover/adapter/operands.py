import polars as pl
from prover.adapter.instruction import (
    DST_BASE_FP,
    OFFSET0,
    OFFSET1,
    OFFSET2,
    OP0_BASE_FP,
    OP1_BASE_AP,
    OP1_BASE_FP,
    OP1_IMM,
)
from prover.adapter.trace import AP, FP, PC

_COL_OP0_BASE = "op0_base"
_COL_OP0_ADDR = "op0_addr"
_COL_OP0 = "op0"
_COL_OP1_BASE = "op1_base"
_COL_OP1_ADDR = "op1_addr"
_COL_OP1 = "op1"
_COL_DST_BASE = "dst_base"
_COL_DST_ADDR = "dst_addr"
_COL_DST = "dst"

OP0_BASE = pl.when(OP0_BASE_FP).then(FP).otherwise(AP).alias(_COL_OP0_BASE)
OP0_ADDR = (OP0_BASE + OFFSET1).alias(_COL_OP0_ADDR)
OP0 = pl.col(_COL_OP0)
OP1_BASE = (
    pl.when(OP1_BASE_FP)
    .then(FP)
    .when(OP1_BASE_AP)
    .then(AP)
    .when(OP1_IMM)
    .then(PC + 1)
    .otherwise(OP0)
    .cast(pl.UInt32)
    .alias(_COL_OP1_BASE)
)
OP1_ADDR = (OP1_BASE + OFFSET2).alias(_COL_OP1_ADDR)
OP1 = pl.col(_COL_OP1)
DST_BASE = pl.when(DST_BASE_FP).then(FP).otherwise(AP).alias(_COL_DST_BASE)
DST_ADDR = (DST_BASE + OFFSET0).alias(_COL_DST_ADDR)
DST = pl.col(_COL_DST)
