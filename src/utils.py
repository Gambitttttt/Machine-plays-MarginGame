import os 
from os.path import join as pjoin
import json 
from enum import Enum
import typing as t

# from configs.game_config import GAME_CONFIG

from src.actions import Action
# from src.players import Player

# def read_game_config() -> t.Dict[str, t.Any]:
#     return json.load(open(pjoin(os.environ['PYTHONPATH'], 'configs/game_config.json'), mode='r', encoding='utf-8'))

class ReadActionType(Enum):
    simple = 'simple'
    full = 'full'


# def read_game_config() -> t.Dict[str, t.Any]:
#     return GAME_CONFIG


def _read_simple_action_from_keyboard():
    print("Input for simple action:")
    field_id, money_invested = input().split(' ')
    return Action(field_id=int(field_id), money_invested=float(money_invested))

def _read_full_action_from_keyboard():
    print("Input for full action:")
    json_ = json.loads(input())
    return Action(**json_)

def read_action_from_keyboard(action_type: ReadActionType=ReadActionType.simple):
    if action_type == ReadActionType.simple:
        return _read_simple_action_from_keyboard()
    else:
        return _read_full_action_from_keyboard()