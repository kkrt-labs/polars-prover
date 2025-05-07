import os
import struct
from pathlib import Path

import polars as pl
from loguru import logger

_COL_AP = "ap"
_COL_FP = "fp"
_COL_PC = "pc"

# Using UInt32 should be enough for the trace, polars will raise in case of overflow
AP = pl.col(_COL_AP).cast(pl.UInt32)
FP = pl.col(_COL_FP).cast(pl.UInt32)
PC = pl.col(_COL_PC).cast(pl.UInt32)

TRACE_SCHEMA = pl.Schema({_COL_AP: pl.UInt32, _COL_FP: pl.UInt32, _COL_PC: pl.UInt32})


def read_trace(file_path: Path) -> pl.LazyFrame:
    record_size = 3 * 8  # 3 registers of 8 bytes
    chunk_size = record_size * 1024 * 1024  # Process 24MB chunks (adjust as needed)

    unpack_format_string = "<3Q"

    chunk_dfs = []

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
                    schema=TRACE_SCHEMA,
                )
                chunk_dfs.append(chunk_df.lazy())  # Append the lazy frame

            iteration += 1

    return pl.concat(chunk_dfs)
