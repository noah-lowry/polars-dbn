import compression.zstd
import concurrent.futures as cf
import os
import struct

import numpy as np
import polars as pl
import polars.selectors as cs
import pyarrow as pa
from polars.io.plugins import register_io_source

from .constants import ACTIONS, RTYPES, SIDES, UNDEF_PRICE, UNDEF_TIMESTAMP
from .dbn_types import Action, RType, Side, schema_num_to_msg_cls


def schema_info_from_number(schema: int, version: int | None = None):
    msg_cls = schema_num_to_msg_cls(schema, version)

    type_layout: np.dtype[np.void] = np.dtype(msg_cls._dtypes)  # ty:ignore[invalid-assignment]
    record_size = msg_cls.size_hint
    fields = msg_cls._ordered_fields
    timestamp_fields = msg_cls._timestamp_fields
    price_fields = msg_cls._price_fields

    polars_schema = {}
    for field in fields:
        if field == "rtype":
            polars_schema[field] = RType
        elif field == "action":
            polars_schema[field] = Action
        elif field == "side":
            polars_schema[field] = Side
        elif field in timestamp_fields:
            polars_schema[field] = pl.Datetime("ns", "UTC")
        elif field in price_fields:
            polars_schema[field] = pl.Decimal(38, 9)
        elif field.endswith("_delta"):
            polars_schema[field] = pl.Duration("ns")
        else:
            polars_schema[field] = pl.datatypes.numpy_char_code_to_dtype(
                type_layout[field]
            )
    polars_schema = pl.Schema(polars_schema)

    return (
        type_layout,
        polars_schema,
        record_size,
        fields,
        timestamp_fields,
        price_fields,
    )


def array_generator(reader, type_layout, is_zstd, batch_size, num_to_read, offset):
    if batch_size is None:
        if is_zstd:
            reader.seek(offset, os.SEEK_SET)
            buffer = np.frombuffer(
                reader.read(
                    num_to_read * type_layout.itemsize
                    if num_to_read is not None
                    else None
                ),
                dtype=type_layout,
            )
        else:
            buffer = np.memmap(reader.name, dtype=type_layout, offset=offset)[
                :num_to_read
            ]
        yield buffer
    else:
        if is_zstd:
            reader.seek(offset, os.SEEK_SET)
            buffer = np.empty(shape=batch_size, dtype=type_layout)
            while True:
                records_read = (
                    reader.readinto(buffer[:num_to_read]) // type_layout.itemsize
                )
                if num_to_read is not None:
                    num_to_read -= records_read
                if records_read == 0:
                    break
                yield buffer[:records_read]
                if records_read < batch_size:
                    break
        else:
            buffer = np.memmap(reader.name, dtype=type_layout, offset=offset)[
                :num_to_read
            ]
            while True:
                batch = buffer[:batch_size]
                buffer = buffer[batch_size:]
                if len(batch) == 0:
                    break
                yield batch
                if len(batch) < batch_size:
                    break


def scan_dbn(path):
    reader = open(path, "rb")
    try:
        is_zstd = reader.peek(4).startswith(b"\x28\xb5\x2f\xfd")
        if is_zstd:
            reader.close()
            reader = compression.zstd.open(path, "rb")

        struct_fmt = "3sbI16xH"
        header_size = struct.calcsize(struct_fmt)
        magic, version, metadata_length, schema_num = struct.unpack(
            struct_fmt, reader.read(header_size)
        )
        if magic != b"DBN":
            raise ValueError("Not a DBN file")
        if schema_num == 65535:
            raise ValueError("Cannot read variable schema")
        data_offset = metadata_length + 8
        (
            type_layout,
            polars_schema,
            record_size,
            fields,
            timestamp_fields,
            price_fields,
        ) = schema_info_from_number(schema_num, version)
    finally:
        reader.close()

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ):
        if is_zstd:
            reader = compression.zstd.open(path, "rb")
        else:
            reader = open(path, "rb")

        try:
            map_fields = fields if with_columns is None else with_columns
            batch_iterator = array_generator(
                reader, type_layout, is_zstd, batch_size, n_rows, data_offset
            )

            for array in batch_iterator:
                with cf.ThreadPoolExecutor(len(map_fields)) as executor:
                    pyarrow_arrays = executor.map(
                        lambda field: pa.array(array[field]), map_fields
                    )
                    df = pl.DataFrame(
                        pa.Table.from_arrays(list(pyarrow_arrays), names=map_fields)
                    )
                df = df.with_columns(
                    # enum types
                    cs.by_name("action", require_all=False)
                    .cast(str)
                    .replace_strict(ACTIONS, default=None, return_dtype=Action),
                    cs.by_name("side", require_all=False)
                    .cast(str)
                    .replace_strict(SIDES, default=None, return_dtype=Side),
                    cs.by_name("rtype", require_all=False)
                    .cast(str)
                    .replace_strict(RTYPES, default=None, return_dtype=RType),
                    # timestamp types
                    cs.by_name(timestamp_fields, require_all=False)
                    .replace(UNDEF_TIMESTAMP, None)
                    .cast(pl.Datetime("ns", time_zone="UTC")),
                    cs.matches(".*_delta$").cast(pl.Duration("ns")),
                    # price types
                    cs.by_name(price_fields, require_all=False)
                    .replace(UNDEF_PRICE, None)
                    .cast(pl.Decimal(38, 9))
                    / 10**9,
                )
                if predicate is not None:
                    df = df.filter(predicate)
                yield df
        finally:
            reader.close()

    return register_io_source(
        source_generator, schema=polars_schema, validate_schema=True
    )


def read_dbn(path):
    return scan_dbn(path).collect()
