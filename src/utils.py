import os 
from os.path import join as pjoin
import json 
from math import inf
from enum import Enum
import typing as t
import random


from actions import Action
# from src.players import Player

def read_game_config(config_path: str) -> t.Dict[str, t.Any]:
    return json.load(open(config_path, mode='r', encoding='utf-8'))

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
    
def init_memory(): # можно попробовать 2 реализации: случайная инициализация из интервала 0,1 и случайная инициализация исходов полей
    # memory = [random.random() for i in range(6)]
    # memory[1] = 0
    memory = [1.1, 0, random.choice([2.1, 0.2]), random.choice([0.0, 1.0, 2.0, 3.0, 4.0]), random.choice([0.7, 1.5, 3.0]), random.choice([2.5, 0.8])]
    return memory