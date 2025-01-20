import typing as t
from actions import Action
from players import Player

PLAYER_ID = int
FIELD_ID = int
Players = t.Dict[PLAYER_ID, Player]
PlayersActions = t.Dict[PLAYER_ID, Action]
PlayersRevenues = t.Dict[PLAYER_ID, float]
FieldsRates = t.Dict[FIELD_ID, float]