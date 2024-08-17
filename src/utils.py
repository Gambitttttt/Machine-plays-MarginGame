import os 
from os.path import join as pjoin
import json 
import typing as t

def read_game_config() -> t.Dict[str, t.Any]:
    return json.load(open(pjoin(os.environ['PYTHONPATH'], 'configs/game_config.json'), mode='r', encoding='utf-8'))