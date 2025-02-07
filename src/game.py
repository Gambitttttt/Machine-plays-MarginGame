import json 
import pprint
import typing as t
from dataclasses import Field, dataclass
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

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

from fields import (
    Fields,
    SberBank,
    CryptoStartup,
    Manufactory,
    OilCompany,
    Profit,
    New_Sector
)
from players import (
    Player
)
from utils import (
    read_game_config,
    ReadActionType,
    read_action_from_keyboard,
)
from constants import (
    PLAYER_ID,
    FIELD_ID,
    Players,
    PlayersActions,
    PlayersRevenues,
    FieldsRates
)

from metrics import(
    basic_metric,
    discounted_metric,
    wins_metric,
    top3_metric,
    mean_rounds_domination
)

from visualizer import(
    one_agent,
    agents_comparison,
    bar_stats
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
    
    def _return_fields_rates(self) -> FieldsRates:
        fields_rates = {}
        for field_id, field in self.fields.items():
            fields_rates[field_id] = field.return_rate()
        return fields_rates

    def request_for_actions(self):
        for player_id, player in self.players.items():
            player.action(num_options=len(self.fields)+1, money_available=player.money)
        
    def recompute_revenues(self):
        players_revenues = self._return_players_revenues()
        for player_id, revenue in players_revenues.items():
            self.players[player_id].money += revenue
            self.players[player_id].cash_history.append(self.players[player_id].money)
            
    def define_winner(self) -> (t.List[PLAYER_ID], float):
        winner_money = [
            player.money
            for player_id, player in self.players.items()
        ]
        top_money = max(winner_money)
        unq_money = sorted(set(winner_money), reverse=True)
        top3_money = [unq_money[0], unq_money[1], unq_money[2]]
        winner_ids = [player_id for player_id, player in self.players.items() if player.money == top_money]
        for winner_id in winner_ids:
            self.players[winner_id].wins += 1
        for player_id, player in self.players.items():
            if player.money in top3_money:
                self.players[player_id].top3 += 1
        return winner_ids, top_money
    
    def define_domination(self, iteration,  end_iteration) -> None:
        winner_money = []
        if iteration == 1:
            for player_id, player in self.players.items():
                self.players[player_id].leading_role = 0
        winner_money = [
            player.money
            for player_id, player in self.players.items()
        ]
        top_money = max(winner_money)
        for player_id, player in self.players.items():
            if player.money == top_money:
                self.players[player_id].leading_role += 1
        if iteration == end_iteration:
            for player_id, player in self.players.items():
                self.players[player_id].domination_rounds.append(self.players[player_id].leading_role)

            
    def print_end_game_results(self):
        print("\n" + '='*50 + 'Final results' + '='*50 + '\n')
        print_players_money(players=self.players)
        winners, top_money = self.define_winner()
        winners_str = ", ".join(f"{color(self.players[winner_id].name, color='magenta')} (player_id: {winner_id})" for winner_id in winners)
        print(f'\nWinner(s): {winners_str} (money: {round(top_money,3)})')
        
        
    def run_game(self) -> t.Dict[int, float]:
        self.init_states()
        pp = pprint.PrettyPrinter(indent=4)
        for i in range(1, self.n_iterations+1):
            print(color(f"\nIteration {i}:", color='green', bold=True))
            self.request_for_actions()
            self.recompute_revenues()
            last_actions = get_players_last_actions(players = self.players)
            self.recompute_state(n_iteration = i, last_actions=last_actions)
            print_players_last_actions(players=self.players)
            print_players_money(players=self.players)
            print()
            #pp.pprint(self.states)
        self.print_end_game_results()
        pp.pprint(self.states)
        print()
        print(f'\nBasic metric for player 1: {basic_metric(dict=self.players, id = '1')}')
        print(f'\nDiscounted metric for player 1: {discounted_metric(dict=self.players, id = '1', discount=0.5)}')

    def run_multiple_games(self, n_games) -> None:
        start_time = time.time()
        print(f'There will be {n_games} games played this time')
        print()
        basic_metric_statistics = {'1': [],
                                   '2': [],
                                   '3': [],
                                   '4': [],
                                   '5': [],
                                   '6': [],
                                   '7': [],
                                   '8': []}
        discounted_metric_statistics = {'1': [],
                                        '2': [],
                                        '3': [],
                                        '4': [],
                                        '5': [],
                                        '6': [],
                                        '7': [],
                                        '8': []}
        for j in range(1, n_games+1):
            for player_id, player in self.players.items():
                self.players[player_id].money = 10
            self.init_states()
            for i in range(1, self.n_iterations+1):
                self.request_for_actions()
                self.recompute_revenues()
                last_actions = get_players_last_actions(players = self.players)
                self.recompute_state(n_iteration = i, last_actions=last_actions)
                self.define_domination(iteration = i, end_iteration=self.n_iterations)
            for player_id, player in self.players.items():
                basic_metric_statistics[str(player_id)].append(basic_metric(dict=self.players, id = player_id))
                discounted_metric_statistics[str(player_id)].append(discounted_metric(dict=self.players, id = player_id, discount=0.5)) 
            winners, top_money = self.define_winner()
        
        end_time = time.time()
        print('Time to complete the program:', end_time - start_time, '\n')
        players_wins = []
        players_top3 = []
        players_round_domination = []
        self.fill_stats_list(stats_list=players_wins, metric=wins_metric, scaling_parameter=n_games)
        self.fill_stats_list(stats_list=players_top3, metric=top3_metric, scaling_parameter=n_games)
        self.fill_stats_list(stats_list=players_round_domination, metric=mean_rounds_domination, scaling_parameter=self.n_iterations)
        
        bar_stats(n_players=8, stats=players_wins, stats_name='Players wins percentage')
        bar_stats(n_players=8, stats=players_top3, stats_name='Players top 3 percentage')
        bar_stats(n_players=8, stats=players_round_domination, stats_name='Players round domination percentage')

        agents_comparison(dict_stats=basic_metric_statistics, xlim=500, n_players=8, metric_type='Basic')
        agents_comparison(dict_stats=discounted_metric_statistics, xlim=500, n_players=8, metric_type='Discounted')
        
        one_agent(dict_basic_stats=basic_metric_statistics, dict_discounted_stats=discounted_metric_statistics, player_id=1)

    def init_states(self):
        self.states = {}
        for i in range(1, self.n_iterations + 1):
            self.states[f'Round {i}'] = {}

    def recompute_state(self, n_iteration, last_actions):
        last_actions_num = [last_actions.count(i) for i in range(1, len(self.fields)+1)]
        rates = self._return_fields_rates()
        for j in range(1, len(self.fields)+1):
            self.states[f'Round {n_iteration}'][f'Field {j}'] = {}
            self.states[f'Round {n_iteration}'][f'Field {j}']['Number of players'] = last_actions_num[0]
            last_actions_num.pop(0)
            self.states[f'Round {n_iteration}'][f'Field {j}']['Return rate'] = rates[j]

    def fill_stats_list(self, stats_list, metric, scaling_parameter):
        if metric == wins_metric:
            text = ['won', 'times or', "% of games"] 
        elif metric == top3_metric:
            text = ['got in top 3', 'times or', "% of games"]
        elif metric == mean_rounds_domination:
            text = ['dominated on average in', 'rounds or', "% of games"]
        for  player_id, player in self.players.items():
            player_stat = metric(dict=self.players, id=player_id)
            stats_list.append(round(player_stat / scaling_parameter * 100,2))
            print(f'Player {player_id} {text[0]} {player_stat} {text[1]} {stats_list[-1]} {text[2]}')
        print() 

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
        n_iterations=game_config['n_iterations']
    )

    return game

def get_players_last_actions(players: Players) -> list:
    last_actions = []
    for i in range(1, len(players)+1):
        last_actions.append(players[i].history[-1].field_id)
    return last_actions

def print_players_last_actions(players: Players) -> None:
    print('\nPlayers last actions:')
    for player_id, player in players.items():
        print(f"\t`{color(player.name, color='magenta')}` (player_id: {player_id}): {player.get_last_action()}")
        
def print_players_money(players: Players) -> None:
    print(color('\nPlayers money:', color='yellow'))
    for player_id, player in players.items():
        print(f"\t`{color(player.name, color='magenta')}` (player_id: {player_id}): {round(player.money,3)}")
        
if __name__ == '__main__':
    
    args = parse_args()
    game_config = read_game_config(config_path=args.config_path)
    # game_config = GAME_CONFIG
    # print("\nGame config:")
    # pprint(game_config)
    n_games = 1000

    game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True)
    print('\n' + '='*100)

    if n_games == 1:
        print(f'\nRunning game (total iterations: {game.n_iterations})...')
        game.run_game()
    else:
        game.run_multiple_games(n_games = n_games)