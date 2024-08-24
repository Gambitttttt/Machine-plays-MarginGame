import json 
from pprint import pprint
import typing as t
from dataclasses import Field, dataclass
import argparse

try:
    from fabulous import color as fb_color
    color = lambda text, color='magenta', bold=False: (
        str(getattr(fb_color, 'bold')(getattr(fb_color, color)(text))) if bold 
        else str(getattr(fb_color, color)(text))
    )
except ImportError as e:
    color = lambda text, color='magenta', bold=False: str(text)
    print("Exception raised trying to import fabulous!")
    print(e, end='\n'*2)

from src.fields import (
    Fields,
    SberBank,
    CryptoStartup,
    Manufactory,
    OilCompany
)
from src.players import (
    Player
)
from src.utils import (
    read_game_config,
    ReadActionType,
    read_action_from_keyboard,
)
from src.constants import (
    PLAYER_ID,
    Players,
    PlayersActions,
    PlayersRevenues
)

def parse_args():
    parser = argparse.ArgumentParser(description='A simple command-line argument parser')
    parser.add_argument(
        '--config_path', 
        type=str, 
        help='Path to the (json) config of the game',
        default='./configs/game_config.json'
    )
    args = parser.parse_args()
    return args
class MarginGame:
    def __init__(
        self, 
        fields: Fields,
        players: Players,
        n_iterations: int=15
    ):
        self.fields = fields
        self.players = players
        self.n_iterations = n_iterations
        
    def _get_last_players_actions(self) -> PlayersActions:
        players_last_actions = {}
        for player_id, player in self.players.items():
            players_last_actions[player_id] = player.get_last_action()
        return players_last_actions
    
    def _return_players_revenues(self) -> PlayersRevenues:
        players_revenues = {}
        for field_id, field in self.fields.items():
            current_players_revenues = field.return_revenues(
                players=self.players
            )
            for player_id, revenue in current_players_revenues.items():
                if player_id not in players_revenues:
                    players_revenues[player_id] = 0
                players_revenues[player_id] += revenue
        return players_revenues
    
    def request_for_actions(self):
        for player_id, player in self.players.items():
            player.action()
        
    def recompute_revenues(self):
        players_revenues = self._return_players_revenues()
        for player_id, revenue in players_revenues.items():
            self.players[player_id].money = revenue
            
    def define_winner(self) -> (t.List[PLAYER_ID], float):
        winner_ids_and_money = [
            (player_id, player.money)
            for player_id, player in self.players.items()
        ]
        winner_ids_and_money = sorted(winner_ids_and_money, key=lambda x: x[1], reverse=True)
        top_money = winner_ids_and_money[0][1]
        winner_ids = [player_id for player_id, player in self.players.items() if player.money == top_money]
        return winner_ids, top_money
            
            
    def print_end_game_results(self):
        print("\n" + '='*50 + 'Final results' + '='*50 + '\n')
        print_players_money(players=self.players)
        winners, top_money = self.define_winner()
        winners_str = ", ".join(f"{color(self.players[winner_id].name, color='magenta')} (player_id: {winner_id})" for winner_id in winners)
        print(f'\nWinner(s): {winners_str} (money: {top_money})')
        
        
    def run_game(self) -> t.Dict[int, float]:
        for i in range(1, self.n_iterations+1):
            print(color(f"\nIteration {i}:", color='green', bold=True))
            self.request_for_actions()
            self.recompute_revenues()
            print_players_last_actions(players=self.players)
            print_players_money(players=self.players)

        self.print_end_game_results()

def initialize_game(
    game_class: MarginGame, 
    game_config: t.Dict[str, t.Any],
    verbose: bool=False
) -> MarginGame:
    # players = game_config['players']
    players = {
        int(player_id): Player.from_dict(player_config)
        for player_id, player_config in game_config['players'].items()
    }
    if verbose: 
        print(color("\nInitialized players:", color='green'))
        for player_id, player in players.items():
            print(f"`{color(player.name, color='magenta')}` (player_id: {player_id})")
        
    # fields = game_config['fields']
    fields = {
        int(field_id): eval(field_config)
        for field_id, field_config in game_config['fields'].items()
    }
    if verbose: 
        print(color("\nInitialized fields:", color='green'))
        for field_id, field in fields.items():
            print(f"{color(field.name, color='magenta')} (field_id: {field_id})")
            print(field.description, end='\n')
    
    print(f"\nfor this game we have {color(game_config['n_iterations'], color='yellow')} iterations")
    
    game = game_class(
        players=players,
        fields=fields,
        n_iterations=game_config['n_iterations'],
    )
    return game 
        
def print_players_last_actions(players: Players) -> None:
    print('\nPlayers last actions:')
    for player_id, player in players.items():
        print(f"\t`{color(player.name, color='magenta')}` (player_id: {player_id}): {player.get_last_action()}")
        
def print_players_money(players: Players) -> None:
    print(color('\nPlayers money:', color='yellow'))
    for player_id, player in players.items():
        print(f"\t`{color(player.name, color='magenta')}` (player_id: {player_id}): {player.money}")
        
if __name__ == '__main__':
    
    args = parse_args()
    game_config = read_game_config(config_path=args.config_path)
    # game_config = GAME_CONFIG
    # print("\nGame config:")
    # pprint(game_config)
    
    game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True)
    print('\n' + '='*100)

    print(f'\nRunning game (total iterations: {game.n_iterations})...')
    game.run_game()    
    
    
    
    

    