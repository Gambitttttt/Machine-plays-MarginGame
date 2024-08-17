import json 
from pprint import pprint
import typing as t
from dataclasses import dataclass

from src.fields import (
    Field
)
from src.players import (
    Player
)
from src.utils import read_game_config


class MarginGame:
    def __init__(
        self, 
        # fields: t.List[Field],
        players: t.List[Player]
    ):
        # self.fields = fields
        self.players = players
        
        
if __name__ == '__main__':
    
    game_config = read_game_config()
    # print("game_config:")
    # pprint(game_config)
    
    game = MarginGame(
        players=[
            Player.from_dict(player_config_dict)
            for player_config_dict in game_config['players']
        ]
    )
    print("\ngame:")
    print(game)
    
    for player in game.players:
        print(player)
    