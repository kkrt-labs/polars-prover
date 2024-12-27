"""
See https://github.com/starkware-libs/stwo-cairo/blob/main/stwo_cairo_prover/crates/adapter/src/memory.rs

The whole purpose of this module is to convert the initial memory from a regular Cairo run, i.e. a Dict[u64, Felt252]
into a memory efficient representation, where each unique observed value of the memory is associated an id so that
the memory can be compressed Dict[u64, u32] and ids can be mapped back to values Dict[u32, Felt252].
"""

# %% Imports
import struct

import polars as pl

# %% Read memory
file_path = "memory.bin"
record_size = 40  # 8 bytes address + 32 bytes value
chunk_size = 40 * 1024 * 1024  # Process 40MB chunks (adjust as needed)

unpack_format_string = "<5Q"
pack_value_format_string = "<4Q"

chunk_dfs = []
schema = pl.Schema({"address": pl.UInt64, "value_bytes": pl.Binary})

iteration = 0

with open(file_path, "rb") as f:
    while True:
        chunk = f.read(chunk_size)
        if iteration > 10:
            break
        if not chunk:
            break

        # Process records within the chunk
        chunk_addresses = []
        chunk_value_bytes = []  # Store packed bytes directly
        for i in range(0, len(chunk), record_size):
            # Ensure we don't read past the end of a partial chunk
            if i + record_size > len(chunk):
                continue  # Or handle partial record if necessary

            unpacked_data = struct.unpack(
                unpack_format_string, chunk[i : i + record_size]
            )
            address = unpacked_data[0]
            value_tuple = unpacked_data[1:]  # Tuple of 4 u64s

            chunk_addresses.append(address)
            # Pack the value tuple back into bytes
            packed_value = struct.pack(pack_value_format_string, *value_tuple)
            chunk_value_bytes.append(packed_value)

        # Create a DataFrame for this chunk
        if chunk_addresses:  # Avoid creating empty DataFrames
            chunk_df = pl.DataFrame(
                {"address": chunk_addresses, "value_bytes": chunk_value_bytes},
                schema=schema,
            )
            chunk_dfs.append(chunk_df.lazy())  # Append the lazy frame

        iteration += 1
        print(f"Processed {iteration} chunks")

# %% Pack memory
lazy_memory = pl.concat(chunk_dfs)

unique_values_df = (
    lazy_memory.select("value_bytes")
    .unique(maintain_order=False)
    .with_row_id()
    .collect(streaming=True)
)

full_memory_with_ids = (
    lazy_memory.join(
        unique_values_df.lazy(),
        on="value_bytes",
        how="inner",
    )
    .select(["address", "id"])
    .rename({"id": "value_id"})
    .collect(streaming=True)
)
