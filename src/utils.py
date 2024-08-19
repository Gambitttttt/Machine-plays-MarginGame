import os 
from os.path import join as pjoin
import json 
from math import inf
from enum import Enum
import typing as t

# from configs.game_config import GAME_CONFIG

from src.actions import Action
# from src.players import Player

def read_game_config() -> t.Dict[str, t.Any]:
    return json.load(open(pjoin(os.environ['PYTHONPATH'], 'configs/game_config.json'), mode='r', encoding='utf-8'))

class ReadActionType(Enum):
    simple = 'simple'
    full = 'full'
    all_money_to_field = 'all_money_to_field'


def _read_field_from_keyboard():
    print("Input format: `<field_id>`")
    field_id = int(input())
    return Action(field_id=int(field_id), money_invested=float(inf))

def _read_simple_action_from_keyboard():
    print("Input format: `<field_id> <money_invested>`")
    field_id, money_invested = input().split(' ')
    return Action(field_id=int(field_id), money_invested=float(money_invested))

def _read_full_action_from_keyboard():
    print('Input format: `{"field_id": ..., "money_invested": ...}`')
    json_ = json.loads(input())
    return Action(**json_)

def read_action_from_keyboard(
    action_type: t.Union[str, ReadActionType]=ReadActionType.simple
):
    action_type_str = (action_type.value if isinstance(action_type, ReadActionType) else action_type)
    if action_type_str == ReadActionType.simple.value:
        return _read_simple_action_from_keyboard()
    elif action_type_str == ReadActionType.full.value:
        return _read_full_action_from_keyboard()
    elif action_type_str == ReadActionType.all_money_to_field.value:
        return _read_field_from_keyboard()
    else:
        raise NotImplementedError