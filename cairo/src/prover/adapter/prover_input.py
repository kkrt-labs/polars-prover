# %% Imports
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from prover.adapter.instruction import decode, opcode
from prover.adapter.memory import read_memory
from prover.adapter.trace import read_trace

load_dotenv()
base_path = Path(os.environ["BASE_PATH"])

# %% Read memory
file_path = base_path / "memory.bin"
memory = read_memory(file_path)

# %% Read trace
file_path = base_path / "trace.bin"
trace = read_trace(file_path)

# %% Build memory dataframes
memory_id_to_value = (
    memory.select("value").unique(maintain_order=False).with_row_index()
)

memory_address_to_id = (
    memory.join(memory_id_to_value, on="value", how="inner")
    .select(["address", "index"])
    .rename({"index": "id"})
)

# %% Prover input
state_transitions = (
    trace.join(
        memory.with_columns(
            encoded_instruction=pl.col("value").struct.field("limb_0")
        ).drop("value"),
        left_on="pc",
        right_on="address",
        how="left",
    )
    .with_columns(
        instruction=pl.col("encoded_instruction").map_batches(
            decode,
            is_elementwise=True,
            return_dtype=pl.Struct(
                {
                    "offset0": pl.Int16,
                    "offset1": pl.Int16,
                    "offset2": pl.Int16,
                    "dst_base_fp": pl.Boolean,
                    "op0_base_fp": pl.Boolean,
                    "op_1_imm": pl.Boolean,
                    "op_1_base_fp": pl.Boolean,
                    "op_1_base_ap": pl.Boolean,
                    "res_add": pl.Boolean,
                    "res_mul": pl.Boolean,
                    "pc_update_jump": pl.Boolean,
                    "pc_update_jump_rel": pl.Boolean,
                    "pc_update_jnz": pl.Boolean,
                    "ap_update_add": pl.Boolean,
                    "ap_update_add_1": pl.Boolean,
                    "opcode_call": pl.Boolean,
                    "opcode_ret": pl.Boolean,
                    "opcode_assert_eq": pl.Boolean,
                    "opcode_extension": pl.UInt8,
                }
            ),
        )
    )
    .with_columns(opcode=opcode(pl.col("instruction")))
    .with_columns(
        dst_addr=(
            pl.when(pl.col("instruction").struct.field("dst_base_fp"))
            .then(pl.col("fp"))
            .otherwise(pl.col("ap"))
        )
    )
    .join(
        memory.rename({"value": "dst"}),
        left_on="dst_addr",
        right_on="address",
        how="left",
    )
    .with_columns(
        opcode=(
            pl.when(pl.col("opcode").eq("opcode_jnz"))
            .then(
                pl.col("opcode")
                + (
                    pl.when(
                        pl.col("dst").eq(
                            pl.lit({"limb_0": 0, "limb_1": 0, "limb_2": 0, "limb_3": 0})
                        )
                    )
                    .then(pl.lit("_taken"))
                    .otherwise(pl.lit(""))
                )
            )
            .otherwise(pl.col("opcode"))
        )
    )
)

instructions_by_pc = state_transitions.unique("pc").select(
    ["pc", "encoded_instruction"]
)

# %% Debug prints
memory_id_to_value.head().collect()
memory_address_to_id.head().collect()
trace.head().collect()
state_transitions.head().collect()
instructions_by_pc.head().collect()
