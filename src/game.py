import json 
import pprint
import typing as t
from dataclasses import field, dataclass
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from collections import deque

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
    init_memory
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
    bar_stats,
    bar_stats_reduced,
    agents_comparison_reduced,
    agents_comparison_uni,
    full_vis
)

from model import(
    DQN,
    trainer,
    Q_table,
    Q_table_trainer
)

from visualizer_train import(
    plot
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
        self.replay_buffer = deque(maxlen=10000)
        
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

    def request_for_actions(self, total_state, model, epsilon=0, n_games=1, decay=0, trained_models=[]):
        if len(trained_models) == 0:
            for player_id, player in self.players.items():
                player.action(num_options=len(self.fields)+1, money_available=player.money, states=self.states, epsilon=epsilon, n_games=n_games, decay=decay, total_state=total_state, model=model)
        else:
            for player_id, player in self.players.items():
                if player_id == 1:
                    player.action(num_options=len(self.fields)+1, money_available=player.money, states=self.states, epsilon=epsilon, n_games=n_games, decay=decay, total_state=total_state, model=model)          
                else:
                    player.action(num_options=len(self.fields)+1, money_available=player.money, states=self.states, epsilon=epsilon, n_games=n_games, decay=decay, total_state=total_state, model=trained_models[player_id-2])      
    
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
        if len(unq_money) > 3:
            top3_money = [unq_money[0], unq_money[1], unq_money[2]]
        else:
            top3_money = unq_money
        winner_ids = [player_id for player_id, player in self.players.items() if player.money == top_money]
        for winner_id in winner_ids:
            self.players[winner_id].wins += 1
        for player_id, player in self.players.items():
            if player.money in top3_money:
                self.players[player_id].top3 += 1
        return winner_ids, top_money, top3_money
    
    def define_domination(self, iteration,  end_iteration):
        winner_money = []
        if iteration == 1:
            for player_id, player in self.players.items():
                self.players[player_id].leading_role = 0
        winner_money = [
            player.money
            for player_id, player in self.players.items()
        ]
        top_money = max(winner_money)
        unq_money = sorted(set(winner_money), reverse=True)
        if len(unq_money) > 3:
            top3_money = [unq_money[0], unq_money[1], unq_money[2]]
        else:
            top3_money = unq_money
            while len(top3_money) != 3:
                top3_money.append(0.0)
        for player_id, player in self.players.items():
            if player.money == top_money:
                self.players[player_id].leading_role += 1
        if iteration == end_iteration:
            for player_id, player in self.players.items():
                self.players[player_id].domination_rounds.append(self.players[player_id].leading_role)
        self.cur_top3_money = top3_money

            
    def print_end_game_results(self):
        print("\n" + '='*50 + 'Final results' + '='*50 + '\n')
        print_players_money(players=self.players)
        winners, top_money, top3_money = self.define_winner()
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
            self.update_memory()
            print_players_last_actions(players=self.players)
            print_players_money(players=self.players)
            print()
            #pp.pprint(self.states)
        self.print_end_game_results()
        pp.pprint(self.states)
        print()
        print(f'\nBasic metric for player 1: {basic_metric(dict=self.players, id = 1)}')
        print(f'\nDiscounted metric for player 1: {discounted_metric(dict=self.players, id = 1, discount=0.5)}')

    def run_training_games_DQN(self, batch_size, epsilon, decay, n_games, trainer, model, trained_models=[]) -> t.Dict[int, float]:
        done=0
        self.init_states()
        for player_id, player in self.players.items():
                self.players[player_id].money = 10
                self.players[player_id].memory = init_memory()
        for i in range(1, self.n_iterations+1):
            old_state = self.return_total_state(turn = i-1, turns_total=self.n_iterations, method='DQN')
            self.request_for_actions(epsilon=epsilon, n_games=n_games,decay=decay, total_state=old_state, model=model, trained_models=trained_models)
            self.recompute_revenues()
            last_actions = get_players_last_actions(players = self.players)
            field_for_training = last_actions[0]
            # print(f'Trainee chose field {field_for_training}')
            action_for_training = [0, 0, 0, 0, 0, 0]
            action_for_training[field_for_training-1]+=1
            self.define_domination(iteration=i, end_iteration=self.n_iterations)
            self.recompute_state(n_iteration = i, last_actions=last_actions)
            new_state = self.return_total_state(turn = i, turns_total=self.n_iterations, method='DQN')
            # reward = self.players[1].money
            reward = self.fields[field_for_training].return_rate()
            # if self.players[1].leading_role == 1:
                # reward *= 1.1
            if i==self.n_iterations:
                done = 1
                # if self.players[1].money in self.cur_top3_money:
                #     reward *= (1 + 0.1 * (self.cur_top3_money.index(self.players[1].money) + 1))
                    # reward += 50 * (self.cur_top3_money.index(self.players[1].money) + 1)
            self.replay_buffer.append((old_state, action_for_training, reward, new_state, done))
            # print(f'Old state: {old_state}')
            # print(f'new state: {new_state}')
            # print(f'Reward: {reward}')
            # print(f'Action: {action_for_training}')
            # print(f'Done: {done}')
            self.train_short_memory(state=old_state, action=last_actions[0], reward=reward, next_state=new_state, trainer=trainer, done=done)
            self.update_memory()
        self.train_long_memory(batch_size=batch_size, trainer=trainer)

    def run_training_games_Q_table(self, epsilon, decay, n_games, trainer, model, trained_models=[]):
        done=0
        self.init_states()
        for player_id, player in self.players.items():
                self.players[player_id].money = 10
                self.players[player_id].memory = init_memory()
        for i in range(1, self.n_iterations+1):
            old_state = self.return_total_state(turn = i-1, turns_total=self.n_iterations, method='Q_table')
            self.request_for_actions(epsilon=epsilon, n_games=n_games, decay=decay, total_state=old_state, model=model, trained_models=trained_models)
            self.recompute_revenues()
            last_actions = get_players_last_actions(players = self.players)
            field_for_training = last_actions[0]
            # print(f'Trainee chose field {field_for_training}')
            action_for_training = field_for_training
            self.define_domination(iteration=i, end_iteration=self.n_iterations)
            self.recompute_state(n_iteration = i, last_actions=last_actions)
            new_state = self.return_total_state(turn = i, turns_total=self.n_iterations, method='Q_table')
            # reward = self.players[1].money
            reward = self.fields[field_for_training].return_rate()
            # if self.players[1].leading_role == 1:
            #     reward *= 1.1
            if i==self.n_iterations:
                done = 1
                # if self.players[1].money in self.cur_top3_money:
                #     reward *= (1 + 0.1 * (self.cur_top3_money.index(self.players[1].money) + 1))
                #     # reward += 50 * (self.cur_top3_money.index(self.players[1].money) + 1)
            trainer.train_step(state=old_state, action=action_for_training, reward=reward, next_state=new_state, done=done)
            self.update_memory()

    def run_multiple_games(self, n_games, model, ls, method) -> None:
        start_time = time.time()
        print(f'There will be {n_games} games played this time')
        print()
        self.basic_metric_statistics = {'1': [],
                                   '2': [],
                                   '3': [],
                                   '4': [],
                                   '5': [],
                                   '6': [],
                                   '7': [],
                                   '8': []}
        self.discounted_metric_statistics = {'1': [],
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
                self.players[player_id].memory = init_memory()
            self.init_states()
            for i in range(1, self.n_iterations+1):
                total_state=self.return_total_state(turn=i-1, turns_total=self.n_iterations, method=method)
                print(f'Total_State: {total_state}')
                self.request_for_actions(total_state=total_state, model=model)
                self.recompute_revenues()
                last_actions = get_players_last_actions(players = self.players)
                print(f'Trained player chooses {last_actions[0]}')
                ls.append(last_actions[0])
                self.define_domination(iteration = i, end_iteration=self.n_iterations)
                self.recompute_state(n_iteration = i, last_actions=last_actions)
                self.update_memory()
            for player_id, player in self.players.items():
                self.basic_metric_statistics[str(player_id)].append(basic_metric(dict=self.players, id = player_id))
                self.discounted_metric_statistics[str(player_id)].append(discounted_metric(dict=self.players, id = player_id, discount=0.5)) 
            winners, top_money, top3_money = self.define_winner()
        
        end_time = time.time()
        print('Time to complete the program:', end_time - start_time, '\n')
        self.players_wins = []
        self.players_top3 = []
        self.players_round_domination = []
        self.fill_stats_list(stats_list=self.players_wins, metric=wins_metric, scaling_parameter=n_games)
        self.fill_stats_list(stats_list=self.players_top3, metric=top3_metric, scaling_parameter=n_games)
        self.fill_stats_list(stats_list=self.players_round_domination, metric=mean_rounds_domination, scaling_parameter=self.n_iterations)
        
        #bar_stats(n_players=8, stats=players_wins, stats_name='Players wins percentage')
        #bar_stats(n_players=8, stats=players_top3, stats_name='Players top 3 percentage')
        #bar_stats(n_players=8, stats=players_round_domination, stats_name='Players round domination percentage')

        # bar_stats_reduced(exp_type='Sber vs Random',
        #                   names=['Sber lover', 'Random'], 
        #                   stats=[players_wins, players_top3, players_round_domination],ids=[1,2], 
        #                   stats_name=['Players wins percentage', 'Players top 3 percentage', 'Players round domination percentage'])
        
        # agents_comparison_reduced(exp_type='Sber vs Random',
        #                           names=['Sber lover', 'Random'],
        #                           ids=[1,2], 
        #                           dict_stats=[basic_metric_statistics,discounted_metric_statistics], 
        #                           xlim=500, 
        #                           metric_types=['Basic', 'Discounted'])
        
        # То, что нужно для визуализации с 1 эпохой
        # full_vis(exp_type='Model vs Memory vs Coop vs Gambler', names=['Model', 'Memory', 'Coop', 'Gambler'], ids=[1,2,4,8], stats=[self.players_wins, self.players_top3, self.players_round_domination],
        #          dict_stats=[self.basic_metric_statistics,self.discounted_metric_statistics], xlim=100000, stats_name=['Players wins percentage', 'Players top 3 percentage', 'Players round domination percentage'],
        #          metric_types=['Basic', 'Discounted'], bar_func=bar_stats_reduced, compare_func=agents_comparison_uni)

        #agents_comparison_reduced(x=1, y=2, ids=[1,2], dict_stats=discounted_metric_statistics, xlim=500, metric_type='Discounted', figname='sber_disc')

        #agents_comparison(dict_stats=basic_metric_statistics, xlim=500, metric_type='Basic')
        #agents_comparison(dict_stats=discounted_metric_statistics, xlim=500, metric_type='Discounted')
        
        #one_agent(dict_basic_stats=basic_metric_statistics, dict_discounted_stats=discounted_metric_statistics, player_id=1)

    def init_states(self):
        self.states = {}
        self.states[1] = {}
        # for i in range(1, self.n_iterations + 1):
        #     self.states[f'Round {i}'] = {}
        #     self.states[i] = {}

    def recompute_state(self, n_iteration, last_actions):
        last_actions_num = [last_actions.count(i) for i in range(1, len(self.fields)+1)]
        rates = self._return_fields_rates()
        # print(rates)
        for j in range(1, len(self.fields)+1):
            # self.states[f'Round {n_iteration}'][f'Field {j}'] = {}
            # self.states[f'Round {n_iteration}'][f'Field {j}']['Number of players'] = last_actions_num[0]
            # last_actions_num.pop(0)
            # self.states[f'Round {n_iteration}'][f'Field {j}']['Return rate'] = rates[j]
            self.states[n_iteration][f'Field {j}'] = {}
            self.states[n_iteration][f'Field {j}']['Number of players'] = last_actions_num[0]
            last_actions_num.pop(0)
            self.states[n_iteration][f'Field {j}']['Return rate'] = rates[j]
            self.states[n_iteration]['Top3 money'] = self.cur_top3_money
        # print(self.states[list(self.states.keys())[-1]])    
        if n_iteration != self.n_iterations:
            self.states[n_iteration+1] = {}


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

    def update_memory(self) -> None:
        rates = self._return_fields_rates()
        for player_id, player in self.players.items():
            last_field = player.get_last_action().field_id
            player.memory[last_field-1] = rates[last_field]

    def return_total_state(self, turn, turns_total, method):
        if turn == 0:
            # if method == 'Q_table':
            #     total_state = [0, 0, 0, 0]    
            # # total_state = [1/turns_total, 0, 0, 0, 0, 0, 0]
            # # total_state = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0] # ход + все поля + деньги + история всех полей
            # # total_state = [0, 0, 0, 0, 0, 0, 0] # ход + все поля
            # # total_state = [0, 0, 0, 0] # ход + кооп поля
            # # total_state = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0] # ход + кооп поля + деньги + история кооп полей
            # elif method == 'DQN':
                # total_state = [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0] # ход + индикаторы хода + кооп поля + деньги + история кооп полей
                # total_state = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0] # ход + кооп поля + деньги + история кооп полей
                #            turn last   last last 3 last    aggreg last            
            total_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # ход + все поля
        elif turn == 1:
            total_state = [turn/turns_total]
            # turn_start_idx = 1 if turn in [0, 1, 2] else 0
            # turn_end_idx = 1 if turn in [12, 13, 14] else 0
            # turns_indices=[turn_start_idx, turn_end_idx]
            prev_turn_state = self.states[turn]
            for item in prev_turn_state.keys():
                if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']: 
                    total_state.append(prev_turn_state[item]['Number of players']/len(self.players.keys()))
            total_state.extend([0, 0, 0, 0, 0, 0])
        elif turn == 2:
            total_state = [turn/turns_total]
            prev_turn_state = self.states[turn]
            for item in prev_turn_state.keys():
                if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']: 
                    total_state.append(prev_turn_state[item]['Number of players']/len(self.players.keys()))
            prev_turn_state = self.states[turn-1]
            for item in prev_turn_state.keys():
                if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']: 
                    total_state.append(prev_turn_state[item]['Number of players']/len(self.players.keys()))
            total_state.extend([0, 0, 0])
        elif turn >= 3:
            total_state = [turn/turns_total]
            prev_turn_state = self.states[turn]
            for item in prev_turn_state.keys():
                if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']: 
                    total_state.append(prev_turn_state[item]['Number of players']/len(self.players.keys()))
            prev_turn_state = self.states[turn-1]
            for item in prev_turn_state.keys():
                if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']: 
                    total_state.append(prev_turn_state[item]['Number of players']/len(self.players.keys()))
            prev_turn_state = self.states[turn-2]
            for item in prev_turn_state.keys():
                if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']: 
                    total_state.append(prev_turn_state[item]['Number of players']/len(self.players.keys()))
        if turn >= 1:
            aggregated_state = [0, 0, 0]
            for idx in range(1, turn+1):
                turn_state = self.states[idx]
                for item in turn_state.keys():
                    if not item in ['Top3 money', 'Field 1', 'Field 2', 'Field 5']:
                        if item == 'Field 3':
                            aggregated_state[0] += turn_state[item]['Number of players']/len(self.players.keys())
                        elif item == 'Field 4':
                            aggregated_state[1] += turn_state[item]['Number of players']/len(self.players.keys())
                        elif item == 'Field 6':
                            aggregated_state[2] += turn_state[item]['Number of players']/len(self.players.keys())
            aggregated_state = np.array(aggregated_state)
            aggregated_state = aggregated_state / turn
            #top3_money = prev_turn_state['Top3 money']
            #player1_money_lead = []
            # for i in range(len(top3_money)):
            #     if self.players[1].money == 0 and top3_money[i] == 0:
            #         player1_money_lead.append(1)
            #     elif top3_money[i] == 0:
            #         player1_money_lead.append(100)
            #     elif self.players[1].money / top3_money[i] > 100:
            #         player1_money_lead.append(100)
            #     else:
            #         player1_money_lead.append(self.players[1].money / top3_money[i])
            # if method == 'DQN':
                # total_state.extend(turns_indices)
                # total_state.extend(player1_money_lead)
            total_state.extend(aggregated_state)
        # print(total_state)
        if not method in ['DQN', 'Q_table']:
            total_state = []
        return list(np.array(total_state, dtype = float))
    
    def train_long_memory(self, batch_size, trainer):
        if len(self.replay_buffer) > batch_size:
            mini_sample = random.sample(self.replay_buffer, batch_size)
        else:
            mini_sample = self.replay_buffer
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, trainer, done):
        trainer.train_step(state, action, reward, next_state, done)


def initialize_game(
    game_class: MarginGame, 
    game_config: t.Dict[str, t.Any],
    verbose: bool=False,
    classes: str='original',
    method: str='DQN',
    Q_table_models: t.List[str]=field(default_factory=list),
    DQN_models: t.List[str]=field(default_factory=list)
) -> MarginGame:
    # players = game_config['players']
    players = {
            int(player_id): Player.from_dict(player_config)
            for player_id, player_config in game_config['players'].items()
    }
    if classes == 'random':
        for id in players.keys():
            method_type = 'custom_all_in'
            players[id].method_type = method_type
            action_type = random.choice(['sber_lover', 'lottery_man', 'manufacturer', 'oil_lover', 'gambler', 'cooperator', 'coop_based', 'memory_based'])
            players[id].action_type = action_type
    elif classes == 'trained':
        for id in players.keys():
            players[id].method_type = 'custom_all_in'
            if id == 1:
                if method == 'DQN':
                    players[id].action_type = 'DQN_learning'
                elif method == 'Q_table':
                    players[id].action_type = 'Q_table_learning'
            else:
                if method == 'DQN':
                    players[id].action_type = 'DQN'
                elif method == 'Q_table':
                    players[id].action_type = 'Q_table'
    elif classes == 'assessing_trained':
        # print(100)
        for id in players.keys():
            players[id].method_type = 'custom_all_in'
            if id == 1:
                players[id].action_type = method
            else:
                action_type = random.choice(['sber_lover', 'lottery_man', 'manufacturer', 'oil_lover', 'gambler', 'cooperator', 'coop_based', 'memory_based'])
                # action_type = random.choices(['sber_lover', 'lottery_man', 'manufacturer', 'oil_lover', 'gambler', 'cooperator', 'coop_based', 'memory_based', 'DQN'], weights=[1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/2])[0] # Пока без DQN
                players[id].action_type = action_type
                if action_type == 'Q_table':
                    # print('!!!')
                    players[id].model_name = random.choice(Q_table_models)
                    model = Q_table(num_states=11000, num_actions=6)
                    players[id].model = model
                    players[id].model = model.load(file_name=players[id].model_name)  # ???
                elif action_type == 'DQN':
                    # print('???')
                    players[id].model_name = random.choice(DQN_models)
                    model = DQN()
                    players[id].model = model
                    players[id].model.load(file_name=players[id].model_name)
                    # print(players[id].model) #???
                # print(players[id].model_name)
                
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
        
def autonomous_game(n_games, epochs, classes, model, model_name, method, Q_table_models=[], DQN_models=[]):
    player1_actions = []
    customs_stats = {
        'sber_lover': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'lottery_man': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'manufacturer': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'oil_lover': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                      'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'gambler': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                    'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'cooperator': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                       'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'coop_based': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                       'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'memory_based': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'Q_table': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'DQN': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []},
        'Main': {'wins': [], 'top3': [], 'domination_rounds': [], 'basic metric q1': [], 'basic metric q2': [],
                         'basic metric q3': [], 'discounted metric q1': [], 'discounted metric q2': [], 'discounted metric q3': []}                                                   
    }
    model.load(file_name=model_name)
    args = parse_args()
    game_config = read_game_config(config_path=args.config_path)
    # game_config = GAME_CONFIG
    # print("\nGame config:")
    # pprint(game_config)
    n_games = n_games
    epochs = epochs

    for j in range(epochs):
        game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes=classes, method = method, Q_table_models=Q_table_models, DQN_models=DQN_models)
        print('\n' + '='*100)
        game.players[1].model_name=model_name
        # if n_games == 1:
        #     print(f'\nRunning game (total iterations: {game.n_iterations})...')
        #     game.run_game(model=model)
        # else:
        game.run_multiple_games(n_games = n_games, model=model, ls = player1_actions, method=method)
        if epochs != 1:
            for player_id, player in game.players.items():
                if game.players[player_id] == game.players[1]:
                    dict = customs_stats['Main']
                else:
                    player = game.players[player_id]   
                    dict = customs_stats[player.action_type]
                dict['wins'].append(game.players_wins[player_id-1])
                dict['top3'].append(game.players_top3[player_id-1])
                dict['domination_rounds'].append(game.players_round_domination[player_id-1])
                basic_stats=game.basic_metric_statistics[str(player_id)]
                dict['basic metric q1'].append(np.quantile(basic_stats, 0.25))
                dict['basic metric q2'].append(np.quantile(basic_stats, 0.5))
                dict['basic metric q3'].append(np.quantile(basic_stats, 0.75))
                discounted_stats=game.discounted_metric_statistics[str(player_id)]
                dict['discounted metric q1'].append(np.quantile(discounted_stats, 0.25))
                dict['discounted metric q2'].append(np.quantile(discounted_stats, 0.5))
                dict['discounted metric q3'].append(np.quantile(discounted_stats, 0.75))
    for i in range(1, 7):
        print(f'Trained player chose {i} {player1_actions.count(i)} times')
    if epochs != 1:
        file_path='stats.json'
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(customs_stats, file, ensure_ascii=False, indent=4)

def train_with_DQN(self_play):
    BATCH_SIZE = 128
    LR = 0.1
    EPSILON = 1
    GAMMA = 0.99
    DECAY = 800000

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    top_score = 0

    args = parse_args()
    game_config = read_game_config(config_path=args.config_path)

    model = DQN()
    target_model = DQN()   
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    DQN_trainer = trainer(model=model, lr=LR, gamma=GAMMA, target_model=target_model)

    if not self_play:
        n_games=1
        while True:
            if n_games % 1000 == 0:
                target_model.load_state_dict(model.state_dict())
                target_model.eval()
                DQN_trainer = trainer(model=model, lr=LR, gamma=GAMMA, target_model=target_model)

            game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='original', method='DQN')
            game.run_training_games_DQN(batch_size=BATCH_SIZE, epsilon=EPSILON, decay=DECAY, n_games=n_games, trainer=DQN_trainer, model=model, trained_models=[])

            score=basic_metric(dict=game.players, id = 1)
            
            if score > top_score:
                    top_score = score
                    model.save(file_name='DQN_top_mid-train.pth')

            print('Game', n_games, 'Score', score, 'Top score:', top_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            if n_games % 40000 == 0:
                model.save(file_name=f'DQN_{n_games}_without_self-play_mid-train.pth')
            n_games+=1
    else:
        SAVE_STEPS=60000
        POLICY_BUFFER_LEN=10
        PLAY_VS_LATEST_POLICY_RATIO=0.5
        SWAP_STEPS=60000
        model_folder = deque(maxlen=POLICY_BUFFER_LEN)
        game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='original', method='DQN')
        n_games=1
        while True:
            if n_games % 1000 == 0:
                target_model.load_state_dict(model.state_dict())
                target_model.eval()
                DQN_trainer = trainer(model=model, lr=LR, gamma=GAMMA, target_model=target_model)    

            if n_games % SAVE_STEPS == 0:
                name=f'DQN_{n_games}_self_play_mid-train.pth'
                model.save(file_name=name)
                additional_model = DQN()
                additional_model.load(file_name=name)
                model_folder.append(additional_model)
                # print(model_folder)

            # if n_games < SWAP_STEPS:
            #     game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='original')
            # else:
            #     game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='trained')
            
            if n_games % SWAP_STEPS == 0:
                if len(model_folder) == 1:
                    models_samples = [model_folder[0]] * 7
                else:
                    models_samples = []
                    for i in range(7): # игроков 8, выбираем модельки для семерых
                        random_num = random.random()
                        if random_num > PLAY_VS_LATEST_POLICY_RATIO:
                            models_samples.append(model_folder[-1])
                        else:
                            random_idx = random.choice(range(len(model_folder)-1))
                            models_samples.append(model_folder[random_idx])
                game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='trained', method='DQN')
            if len(model_folder) == 0:
                models_samples = []

            game.run_training_games_DQN(batch_size=BATCH_SIZE, epsilon=EPSILON, decay=DECAY, n_games=n_games % SAVE_STEPS, trainer=DQN_trainer, model=model, trained_models=models_samples)
            
            score=basic_metric(dict=game.players, id = 1)
            if score > top_score:
                    top_score = score
                    model.save(file_name='DQN_top_mid-train.pth')

            if n_games % 20000 == 0:
                name=f'DQN_{n_games}_self_play_mid-train.pth'
                model.save(file_name=name)

            print('Game', n_games, 'Score', score, 'Top score:', top_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            n_games+=1

def train_with_Q_table(self_play):
    LR = 0.1
    EPSILON = 1
    GAMMA = 0.99
    DECAY = 800000

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    top_score = 0

    args = parse_args()
    game_config = read_game_config(config_path=args.config_path)

    model = Q_table(num_states=11000, num_actions=6)
    trainer_Q_table = Q_table_trainer(model=model, lr=LR, gamma=GAMMA)

    if not self_play:
        n_games=1
        while True:
            game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='original', method='Q_table')
            game.run_training_games_Q_table(epsilon=EPSILON, decay=DECAY, n_games=n_games, trainer=trainer_Q_table, model=model, trained_models=[])

            score=basic_metric(dict=game.players, id = 1)
            
            if score > top_score:
                    top_score = score
                    model.save(name='Q_table_top_without_self_play.npy')

            print('Game', n_games, 'Score', score, 'Top score:', top_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            if n_games % 40000 == 0:
                model.save(name=f'Q_table_without_self_play_{n_games}.npy')
            n_games+=1
    else:
        SAVE_STEPS=80000
        POLICY_BUFFER_LEN=15
        PLAY_VS_LATEST_POLICY_RATIO=0.5
        SWAP_STEPS=80000
        model_folder = deque(maxlen=POLICY_BUFFER_LEN)
        game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='original', method='Q_table')
        n_games=1
        while True:

            if n_games % SAVE_STEPS == 0:
                name=f'Q_table_mid-train_with_self_play_{n_games}.npy'
                model.save(name=name)
                additional_model = Q_table(num_states=11000, num_actions=6)
                additional_model.load(file_name=name)
                model_folder.append(additional_model)

            # if n_games < SWAP_STEPS:
            #     game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='original', method='Q_table')
            # else:
            #     game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='trained', method='Q_table')
            
            if n_games % SWAP_STEPS == 0:
                if len(model_folder) == 1:
                    models_samples = [model_folder[0]] * 7
                else:
                    models_samples = []
                    for i in range(7): # игроков 8, выбираем модельки для семерых
                        random_num = random.random()
                        if random_num > PLAY_VS_LATEST_POLICY_RATIO:
                            models_samples.append(model_folder[-1])
                        else:
                            random_idx = random.choice(range(len(model_folder)-1))
                            models_samples.append(model_folder[random_idx])
                game = initialize_game(game_class=MarginGame, game_config=game_config, verbose=True, classes='trained', method='Q_table')
            if len(model_folder) == 0:
                models_samples = []

            game.run_training_games_Q_table(epsilon=EPSILON, decay=DECAY, n_games=n_games % SWAP_STEPS, trainer=trainer_Q_table, model=model, trained_models=models_samples)
            
            score=basic_metric(dict=game.players, id = 1)
            if score > top_score:
                    top_score = score
                    model.save(name='Q_table_top_with_self_play.npy')

            print('Game', n_games, 'Score', score, 'Top score:', top_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            if n_games % 40000 == 0:
                model.save(name=f'Q_table_with_self_play_{n_games}.npy')
            n_games+=1

if __name__ == '__main__':                                                      #Q_table(num_states=11000, num_actions=6)                        
    # autonomous_game(n_games=20, epochs=200, classes='assessing_trained', model=Q_table(num_states=11000, num_actions=6), model_name='Q_table_without_self_play_600000.npy', method='coop_based',
    #                 Q_table_models=['Q_table_mid-train_with_self_play_80000 (1).npy', 'Q_table_top_with_self_play (5).npy', 'Q_table_without_self_play_40000 (1).npy', 'Q_table_without_self_play_120000.npy'],
    #                 DQN_models=['DQN_top_mid-train.pth', 'DQN_160000_self_play_mid-train (1).pth', 'DQN_60000_self_play_mid-train.pth', 'DQN_120000_self_play_mid-train.pth'])
    # train_with_DQN(self_play=False)
    train_with_Q_table(self_play=False)

    # file_path='stats.json'
    # with open(file_path, "r", encoding="utf-8") as file:
    #     data=json.load(file)
    # path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/Duo_comparison_vis/'
    # for metric in ['wins','top3','domination_rounds','basic metric q1','basic metric q2','basic metric q3','discounted metric q1','discounted metric q2','discounted metric q3']:
    #     for type in data.keys():
    #         if type != 'Main':
    #             plt.clf()
    #             plt.hist(data[type][metric], bins=50, density=True, label=type, alpha=0.5) #density=True, bins=50?
    #             plt.hist(data['Main'][metric], bins=50, density=True, label=type, alpha=0.5) #density=True, bins=50?
    #             plt.legend([type, 'Main'])
    #             plt.ylabel('Относительная частота')
    #             plt.xlabel('Значения метрики')
    #             plt.title(f'Main vs {type}: {metric}')
    #             name=path+f'Main vs {type}_{metric}.png'
    #             plt.savefig(name)
    # print(np.mean(data['Main']['wins']))
    # print(np.mean(data['Main']['top3']))
    # print(np.mean(data['Main']['domination_rounds']))

    # file_path='stats.json'
    # with open(file_path, "r", encoding="utf-8") as file:
    #     data=json.load(file)
    # path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/Overall_vis/'
    # for type in data.keys():
    #     for metric in ['wins','top3','domination_rounds','basic metric q1','basic metric q2','basic metric q3','discounted metric q1','discounted metric q2','discounted metric q3']:
    #         plt.clf()
    #         plt.hist(data[type][metric], bins=50, density=True) #density=True, bins=50?
    #         plt.title(type + ':' + metric)
    #         name=path+type+'_'+metric+'.png'
    #         plt.savefig(name)