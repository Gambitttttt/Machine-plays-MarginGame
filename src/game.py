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
from src.utils import (
    # read_game_config,
    ReadActionType,
    read_action_from_keyboard,
)

from configs.game_config import GAME_CONFIG

class MarginGame:
    def __init__(
        self, 
        fields: t.Dict[int, Field],
        players: t.Dict[int, Player]
    ):
        self.fields = fields
        self.players = players
        self.n_iterations: int=2
        
        
    def run_game(self) -> t.Dict[int, float]:
        for i in range(1, self.n_iterations+1):
            print(f"\nIteration {i}:")
            last_actions = {}
            for player_id, player in self.players.items():
                player.action()
                last_actions[player_id] = player.get_last_action()
            print_players_last_actions(players=self.players)
            
            # computing rewards
            players_rewards = {}
            for field_id, field in self.fields.items():
                current_players_rewards = field.return_revenues( 
                    players_actions=last_actions
                )
                players_rewards.update(current_players_rewards)
                
            print("\nRewards:")
            for player_id, reward in players_rewards.items():
                print(f'player_id: {player_id} reward: {reward}')

def initialize_game(
    game_class: MarginGame, 
    game_config: t.Dict[str, t.Any],
    verbose: bool=False
) -> MarginGame:
    # players={
    #     player_id: Player.from_dict(player_config_dict)
    #     for player_id, player_config_dict in game_config['players'].items()
    # }
    players = game_config['players']
    if verbose: 
        print("\nInitialized players:")
        for player_id, player in players.items():
            print(player)
        
    fields = game_config['fields']
    if verbose: 
        print("\nInitialized fields:")
        for field_id, field in fields.items():
            print(field)
    
    game = game_class(
        players=players,
        fields=fields,
        # n_iterations=game_config['n_iterations'],
    )
    return game 
        
def print_players_last_actions(players: t.Dict[int, Player]) -> None:
    for player_id, player in players.items():
        print(f"\nplayer_id: {player_id}")
        print(player.get_last_action())
        
if __name__ == '__main__':
    
    # game_config = read_game_config()
    game_config = GAME_CONFIG
    print("\nGame config:")
    pprint(game_config)
    
    game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True)
    print('\n' + '='*100)

    print('\nRunning game...')
    game.run_game()    
    
    
    
    

    