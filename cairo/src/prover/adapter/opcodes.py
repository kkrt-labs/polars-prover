import polars as pl
from prover.adapter.instruction import (
    AP_UPDATE_ADD,
    AP_UPDATE_ADD_1,
    DST_BASE_FP,
    OFFSET0,
    OFFSET1,
    OFFSET2,
    OP0_BASE_FP,
    OP_1_BASE_AP,
    OP_1_BASE_FP,
    OP_1_IMM,
    OPCODE_ASSERT_EQ,
    OPCODE_CALL,
    OPCODE_EXTENSION,
    OPCODE_RET,
    PC_UPDATE_JNZ,
    PC_UPDATE_JUMP,
    PC_UPDATE_JUMP_REL,
    RES_ADD,
    RES_MUL,
)

_COL_OPCODE = "opcode"
RET_OPCODE = pl.lit("ret_opcode")
ADD_AP_OPCODE = pl.lit("add_ap_opcode")
JUMP_OPCODE_REL_IMM = pl.lit("jump_opcode_rel_imm")
JUMP_OPCODE_REL = pl.lit("jump_opcode_rel")
JUMP_OPCODE_DOUBLE_DEREF = pl.lit("jump_opcode_double_deref")
JUMP_OPCODE = pl.lit("jump_opcode")
CALL_OPCODE_REL = pl.lit("call_opcode_rel")
CALL_OPCODE_OP_1_BASE_FP = pl.lit("call_opcode_op_1_base_fp")
CALL_OPCODE = pl.lit("call_opcode")
JNZ_OPCODE = pl.lit("jnz_opcode")
JNZ_OPCODE_TAKEN = pl.lit("jnz_opcode_taken")
ASSERT_EQ_OPCODE_IMM = pl.lit("assert_eq_opcode_imm")
ASSERT_EQ_OPCODE_DOUBLE_DEREF = pl.lit("assert_eq_opcode_double_deref")
ASSERT_EQ_OPCODE = pl.lit("assert_eq_opcode")
MUL_OPCODE = pl.lit("mul_opcode")
ADD_OPCODE = pl.lit("add_opcode")
BLAKE_OPCODE = pl.lit("blake_opcode")
QM31_ADD_MUL_OPCODE = pl.lit("qm31_add_mul_opcode")
GENERIC_OPCODE = pl.lit("generic_opcode")

STONE_OPCODE_EXTENSION = 0
BLAKE_OPCODE_EXTENSION = 1
BLAKE_FINALIZE_OPCODE_EXTENSION = 2
QM31_OPCODE_EXTENSION = 3


# ret
_mask_ret = (
    (OFFSET0 == -2)
    & (OFFSET1 == -1)
    & (OFFSET2 == -1)
    & DST_BASE_FP
    & OP0_BASE_FP
    & OP_1_IMM.not_()
    & OP_1_BASE_FP
    & OP_1_BASE_AP.not_()
    & RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JUMP
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & AP_UPDATE_ADD_1.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET
    & OPCODE_ASSERT_EQ.not_()
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
)

# add ap
_mask_add_ap = (
    (OFFSET0 == -1)
    & (OFFSET1 == -1)
    & DST_BASE_FP
    & OP0_BASE_FP
    & RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD
    & AP_UPDATE_ADD_1.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ.not_()
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
    # Only one of op_1_imm, op_1_base_fp, OP_1_BASE_AP must be 1
    & (
        (
            OP_1_IMM.cast(pl.UInt8)
            + OP_1_BASE_FP.cast(pl.UInt8)
            + OP_1_BASE_AP.cast(pl.UInt8)
        )
        == 1
    )
    # If op_1_imm is True, then offset2 must be 1 (next pc)
    & (OP_1_IMM.not_() | (OFFSET2 == 1))
)

# jump
_mask_jump_base = (
    (OFFSET0 == -1)
    & DST_BASE_FP
    & RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ.not_()
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
)

_mask_jump_rel_imm = _mask_jump_base & (
    OP_1_IMM
    & PC_UPDATE_JUMP_REL
    & PC_UPDATE_JUMP.not_()
    & OP_1_BASE_FP.not_()
    & OP_1_BASE_AP.not_()
    & OP0_BASE_FP
    & (OFFSET1 == -1)
    & (OFFSET2 == 1)
)

_mask_jump_rel = _mask_jump_base & (
    OP_1_IMM.not_()
    & PC_UPDATE_JUMP_REL
    & PC_UPDATE_JUMP.not_()
    & (OP_1_BASE_FP | OP_1_BASE_AP)
    & OP0_BASE_FP
    & (OFFSET1 == -1)
)

_mask_jump_double_deref = _mask_jump_base & (
    OP_1_IMM.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & OP_1_BASE_FP.not_()
    & OP_1_BASE_AP.not_()
    & PC_UPDATE_JUMP
)

_mask_jump_abs = _mask_jump_base & (
    OP_1_IMM.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & (OP_1_BASE_FP | OP_1_BASE_AP)
    & OP0_BASE_FP
    & PC_UPDATE_JUMP
    & (OFFSET1 == -1)
)

# call
_mask_call_base = (
    (OFFSET0 == 0)
    & (OFFSET1 == 1)
    & DST_BASE_FP.not_()
    & OP0_BASE_FP.not_()
    & RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & AP_UPDATE_ADD_1.not_()
    & OPCODE_CALL
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ.not_()
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
)

_mask_call_rel = _mask_call_base & (
    PC_UPDATE_JUMP_REL
    & OP_1_IMM
    & OP_1_BASE_FP.not_()
    & OP_1_BASE_AP.not_()
    & (OFFSET2 == 1)
    & PC_UPDATE_JUMP.not_()
)

_mask_call_abs_fp = _mask_call_base & (
    PC_UPDATE_JUMP_REL.not_()
    & OP_1_BASE_FP
    & OP_1_BASE_AP.not_()
    & OP_1_IMM.not_()
    & PC_UPDATE_JUMP
)

_mask_call_abs_ap = _mask_call_base & (
    PC_UPDATE_JUMP_REL.not_() & OP_1_BASE_AP & OP_1_IMM.not_() & PC_UPDATE_JUMP
)

# jnz
_mask_jnz = (
    (OFFSET1 == -1)
    & (OFFSET2 == 1)
    & OP0_BASE_FP
    & OP_1_IMM
    & OP_1_BASE_FP.not_()
    & OP_1_BASE_AP.not_()
    & RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ.not_()
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
)

# assert equal
_mask_assert_eq_base = (
    RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
)

_mask_assert_eq_imm = _mask_assert_eq_base & (
    OP_1_IMM
    & OP_1_BASE_FP.not_()
    & OP_1_BASE_AP.not_()
    & (OFFSET2 == 1)
    & OP0_BASE_FP
    & (OFFSET1 == -1)
)

_mask_assert_eq_double_deref = _mask_assert_eq_base & (
    OP_1_IMM.not_() & OP_1_BASE_FP.not_() & OP_1_BASE_AP.not_()
)

_mask_assert_eq = _mask_assert_eq_base & (
    OP_1_IMM.not_() & (OP_1_BASE_FP | OP_1_BASE_AP) & (OFFSET1 == -1) & OP0_BASE_FP
)

# mul
_mask_mul = (
    RES_ADD.not_()
    & RES_MUL
    & PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
    & (
        (
            OP_1_IMM.cast(pl.UInt8)
            + OP_1_BASE_FP.cast(pl.UInt8)
            + OP_1_BASE_AP.cast(pl.UInt8)
        )
        == 1
    )
    & (OP_1_IMM.not_() | (OFFSET2 == 1))
)

# add
_mask_add = (
    RES_ADD
    & RES_MUL.not_()
    & PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ
    & OPCODE_EXTENSION.eq(STONE_OPCODE_EXTENSION)
    & (
        (
            OP_1_IMM.cast(pl.UInt8)
            + OP_1_BASE_FP.cast(pl.UInt8)
            + OP_1_BASE_AP.cast(pl.UInt8)
        )
        == 1
    )
    & (OP_1_IMM.not_() | (OFFSET2 == 1))
)

# Blake
_mask_blake = (
    OP_1_IMM.not_()
    & (OP_1_BASE_FP | OP_1_BASE_AP)
    & RES_ADD.not_()
    & RES_MUL.not_()
    & PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ.not_()
    & OPCODE_EXTENSION.is_in([BLAKE_OPCODE_EXTENSION, BLAKE_FINALIZE_OPCODE_EXTENSION])
    & ((OP_1_BASE_FP & OP_1_BASE_AP.not_()) | (OP_1_BASE_FP.not_() & OP_1_BASE_AP))
)

# QM31 Add/Mul
_mask_qm31 = (
    PC_UPDATE_JUMP.not_()
    & PC_UPDATE_JUMP_REL.not_()
    & PC_UPDATE_JNZ.not_()
    & AP_UPDATE_ADD.not_()
    & OPCODE_CALL.not_()
    & OPCODE_RET.not_()
    & OPCODE_ASSERT_EQ
    & OPCODE_EXTENSION.eq(QM31_OPCODE_EXTENSION)
    & ((RES_ADD & RES_MUL.not_()) | (RES_ADD.not_() & RES_MUL))
    & (
        (OP_1_IMM & OP_1_BASE_FP.not_() & OP_1_BASE_AP.not_())
        | (OP_1_IMM.not_() & OP_1_BASE_FP & OP_1_BASE_AP.not_())
        | (OP_1_IMM.not_() & OP_1_BASE_FP.not_() & OP_1_BASE_AP)
    )
    & (OP_1_IMM.not_() | (OFFSET2 == 1))
)

OPCODE = (
    pl.when(_mask_ret)
    .then(RET_OPCODE)
    .when(_mask_add_ap)
    .then(ADD_AP_OPCODE)
    .when(_mask_jump_rel_imm)
    .then(JUMP_OPCODE_REL_IMM)
    .when(_mask_jump_rel)
    .then(JUMP_OPCODE_REL)
    .when(_mask_jump_double_deref)
    .then(JUMP_OPCODE_DOUBLE_DEREF)
    .when(_mask_jump_abs)
    .then(JUMP_OPCODE)
    .when(_mask_call_rel)
    .then(CALL_OPCODE_REL)
    .when(_mask_call_abs_fp)
    .then(CALL_OPCODE_OP_1_BASE_FP)
    .when(_mask_call_abs_ap)
    .then(CALL_OPCODE)
    .when(_mask_jnz)
    .then(JNZ_OPCODE)
    .when(_mask_assert_eq_imm)
    .then(ASSERT_EQ_OPCODE_IMM)
    .when(_mask_assert_eq_double_deref)
    .then(ASSERT_EQ_OPCODE_DOUBLE_DEREF)
    .when(_mask_assert_eq)
    .then(ASSERT_EQ_OPCODE)
    .when(_mask_mul)
    .then(MUL_OPCODE)
    .when(_mask_add)
    .then(ADD_OPCODE)
    .when(_mask_blake)
    .then(BLAKE_OPCODE)
    .when(_mask_qm31)
    .then(QM31_ADD_MUL_OPCODE)
    .otherwise(GENERIC_OPCODE)
    .cast(pl.Categorical)
    .alias(_COL_OPCODE)
)
