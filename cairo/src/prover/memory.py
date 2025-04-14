"""
See https://github.com/starkware-libs/stwo-cairo/blob/main/stwo_cairo_prover/crates/adapter/src/memory.rs

The whole purpose of this module is to convert the initial memory from a regular Cairo run, i.e. a Dict[u64, Felt252]
into a memory efficient representation, where each unique observed value of the memory is associated an id so that
the memory can be compressed Dict[u64, u32] and ids can be mapped back to values Dict[u32, Felt252].
"""

# %% Imports
import os
import struct
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from loguru import logger

# %% Read memory
load_dotenv()
base_path = Path(os.environ["BASE_PATH"])
file_path = base_path / "memory.bin"
record_size = 40  # 8 bytes address + 32 bytes value
chunk_size = record_size * 1024 * 1024  # Process 40MB chunks (adjust as needed)
total_size = os.path.getsize(file_path)
logger.info(f"Memory file total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
memory_len = total_size // record_size

unpack_format_string = "<5Q"
pack_value_format_string = "<4Q"

chunk_dfs = []
# Using UInt32 should be enough for the memory, polars will raise in case of overflow
schema = pl.Schema({"address": pl.UInt32, "value_bytes": pl.Binary})

iteration = 0
total_iterations = memory_len // (chunk_size // record_size)
with open(file_path, "rb") as f:
    while True:
        logger.info(f"Processing chunk {iteration}/{total_iterations}")
        chunk = f.read(chunk_size)
        if not chunk:
            break

        chunk_addresses = []
        chunk_value_bytes = []
        for i in range(0, len(chunk), record_size):
            if i + record_size > len(chunk):
                continue

            unpacked_data = struct.unpack(
                unpack_format_string, chunk[i : i + record_size]
            )
            address = unpacked_data[0]
            value_tuple = unpacked_data[1:]

            chunk_addresses.append(address)
            packed_value = struct.pack(pack_value_format_string, *value_tuple)
            chunk_value_bytes.append(packed_value)

        if chunk_addresses:
            chunk_df = pl.DataFrame(
                {"address": chunk_addresses, "value_bytes": chunk_value_bytes},
                schema=schema,
            )
            chunk_dfs.append(chunk_df.lazy())

        iteration += 1

lazy_memory = pl.concat(chunk_dfs)

# %% Pack memory
unique_values_df = (
    lazy_memory.select("value_bytes")
    .unique(maintain_order=False)
    .with_row_index()
    .collect(streaming=True)
)

full_memory_with_ids = (
    lazy_memory.join(
        unique_values_df.lazy(),
        on="value_bytes",
        how="inner",
    )
    .select(["address", "index"])
    .rename({"index": "id"})
    .collect(streaming=True)
)
