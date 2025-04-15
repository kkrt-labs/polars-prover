"""
See also https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/lang/compiler/instruction.py
"""

import numpy as np
import polars as pl

OFFSET_BITS = 16
STONE_OPCODE_EXTENSION = 0
BLAKE_OPCODE_EXTENSION = 1
BLAKE_FINALIZE_OPCODE_EXTENSION = 2
QM31_OPCODE_EXTENSION = 3


def decode(encoded_instruction: pl.Series) -> pl.Series:
    encoded_instruction_np = encoded_instruction.to_numpy()
    offset0 = (encoded_instruction_np & (2**OFFSET_BITS - 1)).astype(np.uint16)
    offset1 = ((encoded_instruction_np >> OFFSET_BITS) & (2**OFFSET_BITS - 1)).astype(
        np.uint16
    )
    offset2 = (
        (encoded_instruction_np >> (2 * OFFSET_BITS)) & (2**OFFSET_BITS - 1)
    ).astype(np.uint16)
    flags = encoded_instruction_np >> (3 * OFFSET_BITS)

    dst_base_fp = (flags & 1).astype(np.bool)
    flags >>= 1
    op0_base_fp = (flags & 1).astype(np.bool)
    flags >>= 1
    op_1_imm = (flags & 1).astype(np.bool)
    flags >>= 1
    op_1_base_fp = (flags & 1).astype(np.bool)
    flags >>= 1
    op_1_base_ap = (flags & 1).astype(np.bool)
    flags >>= 1
    res_add = (flags & 1).astype(np.bool)
    flags >>= 1
    res_mul = (flags & 1).astype(np.bool)
    flags >>= 1
    pc_update_jump = (flags & 1).astype(np.bool)
    flags >>= 1
    pc_update_jump_rel = (flags & 1).astype(np.bool)
    flags >>= 1
    pc_update_jnz = (flags & 1).astype(np.bool)
    flags >>= 1
    ap_update_add = (flags & 1).astype(np.bool)
    flags >>= 1
    ap_update_add_1 = (flags & 1).astype(np.bool)
    flags >>= 1
    opcode_call = (flags & 1).astype(np.bool)
    flags >>= 1
    opcode_ret = (flags & 1).astype(np.bool)
    flags >>= 1
    opcode_assert_eq = (flags & 1).astype(np.bool)
    flags >>= 1
    opcode_extension = flags.astype(np.uint8)

    if (opcode_extension > 3).any():
        raise ValueError(
            f"Invalid opcode extension: {opcode_extension[opcode_extension > 3]}"
        )

    instruction = {
        "offset0": pl.Series(offset0),
        "offset1": pl.Series(offset1),
        "offset2": pl.Series(offset2),
        "dst_base_fp": pl.Series(dst_base_fp),
        "op0_base_fp": pl.Series(op0_base_fp),
        "op_1_imm": pl.Series(op_1_imm),
        "op_1_base_fp": pl.Series(op_1_base_fp),
        "op_1_base_ap": pl.Series(op_1_base_ap),
        "res_add": pl.Series(res_add),
        "res_mul": pl.Series(res_mul),
        "pc_update_jump": pl.Series(pc_update_jump),
        "pc_update_jump_rel": pl.Series(pc_update_jump_rel),
        "pc_update_jnz": pl.Series(pc_update_jnz),
        "ap_update_add": pl.Series(ap_update_add),
        "ap_update_add_1": pl.Series(ap_update_add_1),
        "opcode_call": pl.Series(opcode_call),
        "opcode_ret": pl.Series(opcode_ret),
        "opcode_assert_eq": pl.Series(opcode_assert_eq),
        "opcode_extension": pl.Series(opcode_extension),
    }

    return pl.DataFrame(instruction).to_struct("instruction")


def opcode(instruction: pl.Expr) -> pl.Expr:
    """
    Classifies instructions into different opcode types based on their fields.
    """
    offset0 = instruction.struct.field("offset0")
    offset1 = instruction.struct.field("offset1")
    offset2 = instruction.struct.field("offset2")
    dst_base_fp = instruction.struct.field("dst_base_fp")
    op0_base_fp = instruction.struct.field("op0_base_fp")
    op_1_imm = instruction.struct.field("op_1_imm")
    op_1_base_fp = instruction.struct.field("op_1_base_fp")
    op_1_base_ap = instruction.struct.field("op_1_base_ap")
    res_add = instruction.struct.field("res_add")
    res_mul = instruction.struct.field("res_mul")
    pc_update_jump = instruction.struct.field("pc_update_jump")
    pc_update_jump_rel = instruction.struct.field("pc_update_jump_rel")
    pc_update_jnz = instruction.struct.field("pc_update_jnz")
    ap_update_add = instruction.struct.field("ap_update_add")
    ap_update_add_1 = instruction.struct.field("ap_update_add_1")
    opcode_call = instruction.struct.field("opcode_call")
    opcode_ret = instruction.struct.field("opcode_ret")
    opcode_assert_eq = instruction.struct.field("opcode_assert_eq")
    opcode_extension = instruction.struct.field("opcode_extension")

    # ret
    mask_ret = (
        (offset0 == -2)
        & (offset1 == -1)
        & (offset2 == -1)
        & dst_base_fp
        & op0_base_fp
        & op_1_imm.not_()
        & op_1_base_fp
        & op_1_base_ap.not_()
        & res_add.not_()
        & res_mul.not_()
        & pc_update_jump
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & ap_update_add_1.not_()
        & opcode_call.not_()
        & opcode_ret
        & opcode_assert_eq.not_()
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
    )
    result = pl.when(mask_ret).then(pl.lit("ret_opcode"))

    # add ap
    mask_add_ap = (
        (offset0 == -1)
        & (offset1 == -1)
        & dst_base_fp
        & op0_base_fp
        & res_add.not_()
        & res_mul.not_()
        & pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add
        & ap_update_add_1.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq.not_()
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
        # Only one of op_1_imm, op_1_base_fp, op_1_base_ap must be 1
        & (
            (
                op_1_imm.cast(pl.UInt8)
                + op_1_base_fp.cast(pl.UInt8)
                + op_1_base_ap.cast(pl.UInt8)
            )
            == 1
        )
        # If op_1_imm is True, then offset2 must be 1 (next pc)
        & (op_1_imm.not_() | (offset2 == 1))
    )
    result = result.when(mask_add_ap).then(pl.lit("add_ap_opcode"))

    # jump
    mask_jump_base = (
        (offset0 == -1)
        & dst_base_fp
        & res_add.not_()
        & res_mul.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq.not_()
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
    )
    mask_jump_rel_imm = mask_jump_base & (
        op_1_imm
        & pc_update_jump_rel
        & pc_update_jump.not_()
        & op_1_base_fp.not_()
        & op_1_base_ap.not_()
        & op0_base_fp
        & (offset1 == -1)
        & (offset2 == 1)
    )
    result = result.when(mask_jump_rel_imm).then(pl.lit("jump_opcode_rel_imm"))

    mask_jump_rel = mask_jump_base & (
        op_1_imm.not_()
        & pc_update_jump_rel
        & pc_update_jump.not_()
        & (op_1_base_fp | op_1_base_ap)
        & op0_base_fp
        & (offset1 == -1)
    )
    result = result.when(mask_jump_rel).then(pl.lit("jump_opcode_rel"))

    mask_jump_double_deref = mask_jump_base & (
        op_1_imm.not_()
        & pc_update_jump_rel.not_()
        & op_1_base_fp.not_()
        & op_1_base_ap.not_()
        & pc_update_jump
    )
    result = result.when(mask_jump_double_deref).then(
        pl.lit("jump_opcode_double_deref")
    )

    mask_jump_abs = mask_jump_base & (
        op_1_imm.not_()
        & pc_update_jump_rel.not_()
        & (op_1_base_fp | op_1_base_ap)
        & op0_base_fp
        & pc_update_jump
        & (offset1 == -1)
    )
    result = result.when(mask_jump_abs).then(pl.lit("jump_opcode"))

    # call
    mask_call_base = (
        (offset0 == 0)
        & (offset1 == 1)
        & dst_base_fp.not_()
        & op0_base_fp.not_()
        & res_add.not_()
        & res_mul.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & ap_update_add_1.not_()
        & opcode_call
        & opcode_ret.not_()
        & opcode_assert_eq.not_()
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
    )
    mask_call_rel = mask_call_base & (
        pc_update_jump_rel
        & op_1_imm
        & op_1_base_fp.not_()
        & op_1_base_ap.not_()
        & (offset2 == 1)
        & pc_update_jump.not_()
    )
    result = result.when(mask_call_rel).then(pl.lit("call_opcode_rel"))

    mask_call_abs_fp = mask_call_base & (
        pc_update_jump_rel.not_()
        & op_1_base_fp
        & op_1_base_ap.not_()
        & op_1_imm.not_()
        & pc_update_jump
    )
    result = result.when(mask_call_abs_fp).then(pl.lit("call_opcode_op_1_base_fp"))

    mask_call_abs_ap = mask_call_base & (
        pc_update_jump_rel.not_() & op_1_base_ap & op_1_imm.not_() & pc_update_jump
    )
    result = result.when(mask_call_abs_ap).then(pl.lit("call_opcode"))

    # jnz
    mask_jnz = (
        (offset1 == -1)
        & (offset2 == 1)
        & op0_base_fp
        & op_1_imm
        & op_1_base_fp.not_()
        & op_1_base_ap.not_()
        & res_add.not_()
        & res_mul.not_()
        & pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq.not_()
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
    )
    # Note: For jnz, we can't determine 'taken' without memory access
    # This will be handled in the next step, using the memory table
    result = result.when(mask_jnz).then(pl.lit("jnz_opcode"))

    # assert equal
    mask_assert_eq_base = (
        res_add.not_()
        & res_mul.not_()
        & pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
    )
    mask_assert_eq_imm = mask_assert_eq_base & (
        op_1_imm
        & op_1_base_fp.not_()
        & op_1_base_ap.not_()
        & (offset2 == 1)
        & op0_base_fp
        & (offset1 == -1)
    )
    result = result.when(mask_assert_eq_imm).then(pl.lit("assert_eq_opcode_imm"))

    mask_assert_eq_double_deref = mask_assert_eq_base & (
        op_1_imm.not_() & op_1_base_fp.not_() & op_1_base_ap.not_()
    )
    result = result.when(mask_assert_eq_double_deref).then(
        pl.lit("assert_eq_opcode_double_deref")
    )

    mask_assert_eq = mask_assert_eq_base & (
        op_1_imm.not_() & (op_1_base_fp | op_1_base_ap) & (offset1 == -1) & op0_base_fp
    )
    result = result.when(mask_assert_eq).then(pl.lit("assert_eq_opcode"))

    # mul
    mask_mul = (
        res_add.not_()
        & res_mul
        & pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
        & (
            (
                op_1_imm.cast(pl.UInt8)
                + op_1_base_fp.cast(pl.UInt8)
                + op_1_base_ap.cast(pl.UInt8)
            )
            == 1
        )
        & (op_1_imm.not_() | (offset2 == 1))
    )
    result = result.when(mask_mul).then(pl.lit("mul_opcode"))

    # add
    mask_add = (
        res_add
        & res_mul.not_()
        & pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq
        & opcode_extension.eq(STONE_OPCODE_EXTENSION)
        & (
            (
                op_1_imm.cast(pl.UInt8)
                + op_1_base_fp.cast(pl.UInt8)
                + op_1_base_ap.cast(pl.UInt8)
            )
            == 1
        )
        & (op_1_imm.not_() | (offset2 == 1))
    )
    result = result.when(mask_add).then(pl.lit("add_opcode"))

    # Blake
    mask_blake = (
        op_1_imm.not_()
        & (op_1_base_fp | op_1_base_ap)
        & res_add.not_()
        & res_mul.not_()
        & pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq.not_()
        & opcode_extension.is_in(
            [BLAKE_OPCODE_EXTENSION, BLAKE_FINALIZE_OPCODE_EXTENSION]
        )
        & ((op_1_base_fp & op_1_base_ap.not_()) | (op_1_base_fp.not_() & op_1_base_ap))
    )
    result = result.when(mask_blake).then(pl.lit("blake_opcode"))

    # QM31 Add/Mul
    mask_qm31 = (
        pc_update_jump.not_()
        & pc_update_jump_rel.not_()
        & pc_update_jnz.not_()
        & ap_update_add.not_()
        & opcode_call.not_()
        & opcode_ret.not_()
        & opcode_assert_eq
        & opcode_extension.eq(QM31_OPCODE_EXTENSION)
        & ((res_add & res_mul.not_()) | (res_add.not_() & res_mul))
        & (
            (op_1_imm & op_1_base_fp.not_() & op_1_base_ap.not_())
            | (op_1_imm.not_() & op_1_base_fp & op_1_base_ap.not_())
            | (op_1_imm.not_() & op_1_base_fp.not_() & op_1_base_ap)
        )
        & (op_1_imm.not_() | (offset2 == 1))
    )
    result = result.when(mask_qm31).then(pl.lit("qm31_add_mul_opcode"))

    # Default case for unclassified instructions
    return result.otherwise(pl.lit("generic_opcode")).cast(pl.Categorical)
