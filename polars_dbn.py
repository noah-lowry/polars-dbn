import struct
import numpy as np
import polars as pl
import pyarrow as pa
from polars.io.plugins import register_io_source
from databento.common.dbnstore import DBNStore, FileDataSource, MemoryDataSource
from databento.common.constants import (
    Schema,
    SCHEMA_STRUCT_MAP,
    SCHEMA_STRUCT_MAP_V1,
    SCHEMA_STRUCT_MAP_V2,
)

SCHEMA_VERSIONS = {1: SCHEMA_STRUCT_MAP_V1, 2: SCHEMA_STRUCT_MAP_V2}


def scan_dbn(source):

    if isinstance(source, DBNStore):
        version = source.metadata.version
        metadata_length = source._metadata_length
        schema_id = int(source.schema)

        if isinstance(source._data_source, FileDataSource):
            source = source._data_source.path
        elif isinstance(source._data_source, MemoryDataSource):
            source = source._data_source.__buffer.getbuffer()
    else:
        with pa.input_stream(source, compression="detect", buffer_size=26) as stream:
            version, metadata_length, schema_id = struct.unpack("<3xBL16xH", stream.read(26))

        metadata_length += 8

    schema = SCHEMA_VERSIONS.get(version, SCHEMA_STRUCT_MAP)[Schema.from_int(schema_id)]
    projections = [
        pl.selectors.by_name(schema._price_fields, require_all=False).replace(pl.Int64.max(), None),
        pl.selectors.by_name(schema._timestamp_fields, require_all=False).cast(
            pl.Datetime("ns", "UTC")
        ),
    ]
    numpy_dtype = np.dtype(schema._dtypes)
    schema = pl.Schema(
        [
            (
                (field, pl.datatypes.numpy_char_code_to_dtype(numpy_dtype[field]))
                if field not in schema._timestamp_fields
                else (field, pl.Datetime("ns", "UTC"))
            )
            for field in schema._ordered_fields
        ]
    )

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ):
        if with_columns is None:
            with_columns = schema.names()
        if batch_size is None:
            batch_size = 1024**2

        fields = pa.schema(
            pa.field(
                field,
                type=(
                    pa.from_numpy_dtype(numpy_dtype[field])
                    if numpy_dtype[field].char != "S"
                    else pa.binary(numpy_dtype[field].itemsize)
                ),
            )
            for field in with_columns
        )

        buffer_rows = batch_size if n_rows is None else min(batch_size, n_rows)
        buff = np.empty(buffer_rows, dtype=numpy_dtype)
        with pa.input_stream(source, compression="detect") as stream:
            stream.read(metadata_length)

            while n_rows is None or n_rows > 0:
                if n_rows is not None:
                    batch_size = min(batch_size, n_rows)

                num_bytes_read = stream.readinto(memoryview(buff))
                if num_bytes_read == 0:
                    n_rows = 0

                arr = buff[:num_bytes_read // numpy_dtype.itemsize]
                arrays = [pa.array(arr[field.name], type=field.type) for field in fields]
                table = pa.Table.from_arrays(arrays, schema=fields)
                df = pl.from_arrow(table).with_columns(*projections)

                if n_rows is not None:
                    n_rows -= df.height

                if df.height < batch_size:
                    n_rows = 0

                if predicate is not None:
                    df = df.filter(predicate)

                yield df

    return register_io_source(
        io_source=source_generator, schema=schema, validate_schema=True, is_pure=True
    )
