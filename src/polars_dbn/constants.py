from databento_dbn import Action, RType, Side

UNDEF_TIMESTAMP = 18446744073709551615
UNDEF_PRICE = 9223372036854775807

RTYPES = {rtype.value: rtype.name for rtype in RType.variants()}
ACTIONS = {
    action.value: action.name for action in Action.variants() if action.value != "N"
}
SIDES = {side.value: side.name for side in Side.variants() if side.value != "N"}

__all__ = [UNDEF_TIMESTAMP, UNDEF_PRICE, RTYPES, ACTIONS, SIDES]
