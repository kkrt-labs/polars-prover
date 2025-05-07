# %% Imports
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from prover.adapter.instruction import _COL_ENCODED_INSTRUCTION, DST_BASE_FP, OFFSET0
from prover.adapter.memory import _COL_ADDRESS, _COL_VALUE, VALUE_ZERO, read_memory
from prover.adapter.opcodes import _COL_OPCODE, JNZ_OPCODE, JNZ_OPCODE_TAKEN, OPCODE
from prover.adapter.trace import _COL_PC, AP, FP, read_trace

pl.enable_string_cache()
load_dotenv()
base_path = Path(os.environ["BASE_PATH"])

_COL_DST_ADDR = "dst_addr"
_COL_DST = "dst"
DST_ADDR = pl.col(_COL_DST_ADDR).cast(pl.UInt32)
DST = pl.col(_COL_DST)


# %% Read memory
file_path = base_path / "memory.bin"
memory = read_memory(file_path)

# %% Read trace
file_path = base_path / "trace.bin"
trace = read_trace(file_path)

# %% Prover input
state_transitions = (
    trace.join(
        memory.with_columns(
            pl.col(_COL_VALUE).struct.field("limb_0").alias(_COL_ENCODED_INSTRUCTION)
        ).drop(_COL_VALUE),
        left_on=_COL_PC,
        right_on=_COL_ADDRESS,
        how="left",
    )
    .with_columns(OPCODE, DST_BASE_FP, OFFSET0)
    .with_columns(
        (pl.when(DST_BASE_FP).then(FP).otherwise(AP) + OFFSET0).alias(_COL_DST_ADDR)
    )
    .join(memory, left_on=_COL_DST_ADDR, right_on=_COL_ADDRESS, how="left")
    .rename({_COL_VALUE: _COL_DST})
    .with_columns(
        pl.when(pl.col(_COL_OPCODE).eq(JNZ_OPCODE) & DST.eq(VALUE_ZERO))
        .then(JNZ_OPCODE_TAKEN)
        .otherwise(pl.col(_COL_OPCODE))
        .alias(_COL_OPCODE)
    )
)

# %% Debug prints
instructions_by_pc = state_transitions.unique("pc").select(
    ["pc", "encoded_instruction"]
)
trace.head().collect()
state_transitions.head().collect()
instructions_by_pc.head().collect()
