# %% Imports
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from prover.adapter.instruction import _COL_ENCODED_INSTRUCTION
from prover.adapter.memory import _COL_ADDRESS, _COL_VALUE, read_memory
from prover.adapter.opcodes import (
    _COL_OPCODE,
    ADD_OPCODE,
    JNZ_OPCODE,
    JNZ_OPCODE_TAKEN,
    OPCODE,
)
from prover.adapter.operands import (
    _COL_DST,
    _COL_DST_ADDR,
    _COL_OP0,
    _COL_OP0_ADDR,
    _COL_OP1,
    _COL_OP1_ADDR,
    DST_ADDR,
    OP0_ADDR,
    OP1_ADDR,
)
from prover.adapter.trace import _COL_PC, read_trace
from prover.components.add_opcode_small import ADD_SMALL_OPCODE

pl.enable_string_cache()
load_dotenv()
base_path = Path(os.environ["BASE_PATH"])


# %% Read memory
file_path = base_path / "memory.bin"
memory = read_memory(file_path)

# %% Read trace
file_path = base_path / "trace.bin"
trace = read_trace(file_path)

# %% Prover input
state_transitions = (
    # Join trace with memory to get encoded instruction and opcode
    trace.join(memory, left_on=_COL_PC, right_on=_COL_ADDRESS, how="left")
    .rename({_COL_VALUE: _COL_ENCODED_INSTRUCTION})
    .with_columns(OPCODE)
    # Add op0 to state transitions
    .with_columns(OP0_ADDR)
    .join(memory, left_on=_COL_OP0_ADDR, right_on=_COL_ADDRESS, how="left")
    .rename({_COL_VALUE: _COL_OP0})
    # Add op1 to state transitions
    .with_columns(OP1_ADDR)
    .join(memory, left_on=_COL_OP1_ADDR, right_on=_COL_ADDRESS, how="left")
    .rename({_COL_VALUE: _COL_OP1})
    # Add dst to state transitions
    .with_columns(DST_ADDR)
    .join(memory, left_on=_COL_DST_ADDR, right_on=_COL_ADDRESS, how="left")
    .rename({_COL_VALUE: _COL_DST})
    # Update jnz opcode (taken or not) based on dst
    .with_columns(
        pl.when(pl.col(_COL_OPCODE).eq(JNZ_OPCODE) & pl.col(_COL_DST).eq(0))
        .then(JNZ_OPCODE_TAKEN)
        .otherwise(pl.col(_COL_OPCODE))
        .alias(_COL_OPCODE)
    )
)

# %% Witnesses
add_small_witness = state_transitions.filter(pl.col(_COL_OPCODE).eq(ADD_OPCODE)).select(
    ADD_SMALL_OPCODE
)

# %% Debug prints
state_transitions.collect_schema()
state_transitions = state_transitions.collect()
state_transitions.head()
(
    state_transitions.filter(pl.col(_COL_OPCODE).eq(ADD_OPCODE)).select(
        [*ADD_SMALL_OPCODE, _COL_OPCODE]
    )
)
instructions_by_pc = state_transitions.unique("pc").select(
    ["pc", "encoded_instruction"]
)
trace.head().collect()
state_transitions.head()
instructions_by_pc.head()
