from prover.adapter.instruction import (
    AP_UPDATE_ADD_1,
    DST_BASE_FP,
    OFFSET0,
    OFFSET1,
    OFFSET2,
    OP0_BASE_FP,
    OP1_BASE_FP,
    OP1_IMM,
)
from prover.adapter.operands import DST, DST_BASE, OP0, OP0_BASE, OP1, OP1_BASE
from prover.adapter.trace import AP, FP, PC

ADD_SMALL_OPCODE = [
    PC,
    AP,
    FP,
    OFFSET0,
    OFFSET1,
    OFFSET2,
    DST_BASE_FP,
    OP0_BASE_FP,
    OP1_IMM,
    OP1_BASE_FP,
    AP_UPDATE_ADD_1,
    DST_BASE,
    OP0_BASE,
    OP1_BASE,
    DST,
    OP0,
    OP1,
]
