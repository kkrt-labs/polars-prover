import polars as pl

# %% Column names
_COL_ENCODED_INSTRUCTION = "encoded_instruction"
_COL_PC = "pc"
_COL_AP = "ap"
_COL_FP = "fp"
_COL_OFFSET0 = "offset0"
_COL_OFFSET1 = "offset1"
_COL_OFFSET2 = "offset2"
_COL_DST_BASE_FP = "dst_base_fp"
_COL_OP0_BASE_FP = "op0_base_fp"
_COL_OP1_IMM = "op_1_imm"
_COL_OP1_BASE_FP = "op_1_base_fp"
_COL_OP1_BASE_AP = "op_1_base_ap"
_COL_RES_ADD = "res_add"
_COL_RES_MUL = "res_mul"
_COL_PC_UPDATE_JUMP = "pc_update_jump"
_COL_PC_UPDATE_JUMP_REL = "pc_update_jump_rel"
_COL_PC_UPDATE_JNZ = "pc_update_jnz"
_COL_AP_UPDATE_ADD = "ap_update_add"
_COL_AP_UPDATE_ADD_1 = "ap_update_add_1"
_COL_OPCODE_CALL = "opcode_call"
_COL_OPCODE_RET = "opcode_ret"
_COL_OPCODE_ASSERT_EQ = "opcode_assert_eq"
_COL_OPCODE_EXTENSION = "opcode_extension"


# %% Column expressions
ENCODED_INSTRUCTION = pl.col(_COL_ENCODED_INSTRUCTION)
PC = pl.col(_COL_PC)
AP = pl.col(_COL_AP)
FP = pl.col(_COL_FP)


# %% Decode instruction
OFFSET_BITS = 16
_power_of_2_offset_bits = 2**OFFSET_BITS

OFFSET0 = (
    ((ENCODED_INSTRUCTION & (_power_of_2_offset_bits - 1)).cast(pl.Int32) - 2**15)
    .cast(pl.Int16)
    .alias(_COL_OFFSET0)
)
OFFSET1 = (
    (
        (
            ENCODED_INSTRUCTION.floordiv(_power_of_2_offset_bits)
            & (_power_of_2_offset_bits - 1)
        ).cast(pl.Int32)
        - 2**15
    )
    .cast(pl.Int16)
    .alias(_COL_OFFSET1)
)
OFFSET2 = (
    (
        (
            (ENCODED_INSTRUCTION.floordiv(_power_of_2_offset_bits**2))
            & (_power_of_2_offset_bits - 1)
        ).cast(pl.Int32)
        - 2**15
    )
    .cast(pl.Int16)
    .alias(_COL_OFFSET2)
)

# Initial flags expression
_current_flags_expr = ENCODED_INSTRUCTION.floordiv(
    _power_of_2_offset_bits**3
)  # Equivalent to >> (3 * OFFSET_BITS)

# Sequentially extract flags
# For each flag:
# 1. Perform bitwise AND with 1
# 2. Cast to Boolean
# 3. Update _current_flags_expr by integer dividing by 2 (equivalent to shifting right by 1)

DST_BASE_FP = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_DST_BASE_FP)
_current_flags_expr = _current_flags_expr.floordiv(2)

OP0_BASE_FP = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OP0_BASE_FP)
_current_flags_expr = _current_flags_expr.floordiv(2)

OP1_IMM = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OP1_IMM)
_current_flags_expr = _current_flags_expr.floordiv(2)

OP1_BASE_FP = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OP1_BASE_FP)
_current_flags_expr = _current_flags_expr.floordiv(2)

OP1_BASE_AP = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OP1_BASE_AP)
_current_flags_expr = _current_flags_expr.floordiv(2)

RES_ADD = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_RES_ADD)
_current_flags_expr = _current_flags_expr.floordiv(2)

RES_MUL = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_RES_MUL)
_current_flags_expr = _current_flags_expr.floordiv(2)

PC_UPDATE_JUMP = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_PC_UPDATE_JUMP)
_current_flags_expr = _current_flags_expr.floordiv(2)

PC_UPDATE_JUMP_REL = (
    (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_PC_UPDATE_JUMP_REL)
)
_current_flags_expr = _current_flags_expr.floordiv(2)

PC_UPDATE_JNZ = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_PC_UPDATE_JNZ)
_current_flags_expr = _current_flags_expr.floordiv(2)

AP_UPDATE_ADD = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_AP_UPDATE_ADD)
_current_flags_expr = _current_flags_expr.floordiv(2)

AP_UPDATE_ADD_1 = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_AP_UPDATE_ADD_1)
_current_flags_expr = _current_flags_expr.floordiv(2)

OPCODE_CALL = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OPCODE_CALL)
_current_flags_expr = _current_flags_expr.floordiv(2)

OPCODE_RET = (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OPCODE_RET)
_current_flags_expr = _current_flags_expr.floordiv(2)

OPCODE_ASSERT_EQ = (
    (_current_flags_expr & 1).cast(pl.Boolean).alias(_COL_OPCODE_ASSERT_EQ)
)
_current_flags_expr = _current_flags_expr.floordiv(2)

# The remaining part of _current_flags_expr is the opcode_extension
# Cast to UInt64 to match the INSTRUCTION schema definition
OPCODE_EXTENSION = _current_flags_expr.cast(pl.UInt64).alias(_COL_OPCODE_EXTENSION)
