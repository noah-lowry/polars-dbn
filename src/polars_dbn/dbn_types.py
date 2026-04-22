from databento.common.constants import (
    SCHEMA_STRUCT_MAP,
    SCHEMA_STRUCT_MAP_V1,
    SCHEMA_STRUCT_MAP_V2,
)
from databento_dbn import Schema
from polars.datatypes import Enum

from .constants import ACTIONS, RTYPES, SIDES

RType = Enum(RTYPES.values())
Action = Enum(ACTIONS.values())
Side = Enum(SIDES.values())


def schema_num_to_msg_cls(schema_num, version):
    schema_struct_map = (
        SCHEMA_STRUCT_MAP_V1
        if version == 1
        else SCHEMA_STRUCT_MAP_V2
        if version == 2
        else SCHEMA_STRUCT_MAP
    )
    msg_cls = schema_struct_map[Schema.from_int(schema_num)]
    return msg_cls


__all__ = [RType, Action, Side]
