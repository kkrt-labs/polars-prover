"""
See https://github.com/starkware-libs/stwo-cairo/blob/main/stwo_cairo_prover/crates/adapter/src/vm_import/mod.rs
"""

# %% Imports
import os
import struct
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from loguru import logger

# %% Read trace
load_dotenv()
base_path = Path(os.environ["BASE_PATH"])
file_path = base_path / "trace.bin"
record_size = 24  # 3 registers of 8 bytes
chunk_size = record_size * 1024 * 1024  # Process 24MB chunks (adjust as needed)

unpack_format_string = "<3Q"

chunk_dfs = []
# Using UInt32 should be enough for the trace, polars will raise in case of overflow
schema = pl.Schema({"ap": pl.UInt32, "fp": pl.UInt32, "pc": pl.UInt32})

total_size = os.path.getsize(file_path)
logger.info(f"Trace file total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
trace_len = total_size // record_size

iteration = 0
total_iterations = trace_len // (chunk_size // record_size)
with open(file_path, "rb") as f:
    while True:
        logger.info(f"Processing chunk {iteration}/{total_iterations}")
        chunk = f.read(chunk_size)
        if not chunk:
            break

        # Process records within the chunk
        chunk_ap = []
        chunk_fp = []
        chunk_pc = []
        for i in range(0, len(chunk), record_size):
            # Ensure we don't read past the end of a partial chunk
            if i + record_size > len(chunk):
                continue  # Or handle partial record if necessary

            unpacked_data = struct.unpack(
                unpack_format_string, chunk[i : i + record_size]
            )
            ap, fp, pc = unpacked_data

            chunk_ap.append(ap)
            chunk_fp.append(fp)
            chunk_pc.append(pc)

        # Create a DataFrame for this chunk
        if chunk_ap:  # Avoid creating empty DataFrames
            chunk_df = pl.DataFrame(
                {"ap": chunk_ap, "fp": chunk_fp, "pc": chunk_pc},
                schema=schema,
            )
            chunk_dfs.append(chunk_df.lazy())  # Append the lazy frame

        iteration += 1


lazy_trace = pl.concat(chunk_dfs)
