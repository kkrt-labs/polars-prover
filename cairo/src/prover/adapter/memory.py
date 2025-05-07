import os
import struct
from pathlib import Path

import polars as pl
from loguru import logger

_COL_ADDRESS = "address"
_COL_VALUE = "value"

# Using UInt32 should be enough for the memory, polars will raise in case of overflow
ADDRESS = pl.col(_COL_ADDRESS).cast(pl.UInt32)
VALUE = pl.col(_COL_VALUE).cast(
    pl.Struct(
        {
            "limb_0": pl.UInt64,
            "limb_1": pl.UInt64,
            "limb_2": pl.UInt64,
            "limb_3": pl.UInt64,
        }
    )
)
VALUE_ZERO = pl.lit({"limb_0": 0, "limb_1": 0, "limb_2": 0, "limb_3": 0})

MEMORY_SCHEMA = pl.Schema(
    {
        _COL_ADDRESS: pl.UInt32,
        _COL_VALUE: pl.Struct(
            {
                "limb_0": pl.UInt64,
                "limb_1": pl.UInt64,
                "limb_2": pl.UInt64,
                "limb_3": pl.UInt64,
            }
        ),
    }
)


def read_memory(file_path: Path) -> pl.LazyFrame:

    record_size = 40  # 8 bytes address + 32 bytes value
    chunk_size = record_size * 1024 * 1024  # Process 40MB chunks (adjust as needed)
    total_size = os.path.getsize(file_path)
    logger.info(f"Memory file total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    memory_len = total_size // record_size

    unpack_format_string = "<5Q"

    chunk_dfs = []

    iteration = 0
    total_iterations = memory_len // (chunk_size // record_size)
    with open(file_path, "rb") as f:
        while True:
            logger.info(f"Processing chunk {iteration}/{total_iterations}")
            chunk = f.read(chunk_size)
            if not chunk:
                break

            chunk_addresses = []
            chunk_value = []
            for i in range(0, len(chunk), record_size):
                if i + record_size > len(chunk):
                    continue

                unpacked_data = struct.unpack(
                    unpack_format_string, chunk[i : i + record_size]
                )
                address = unpacked_data[0]
                value_tuple = unpacked_data[1:]

                chunk_addresses.append(address)
                chunk_value.append(value_tuple)

            if chunk_addresses:
                chunk_df = pl.DataFrame(
                    {"address": chunk_addresses, "value": chunk_value},
                    schema=MEMORY_SCHEMA,
                )
                chunk_dfs.append(chunk_df.lazy())

            iteration += 1

    return pl.concat(chunk_dfs)
