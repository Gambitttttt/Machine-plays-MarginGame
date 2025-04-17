import typing as t 
import random
import torch
from math import inf
from dataclasses import dataclass
from dataclasses import field 

try:
    from fabulous import color as fb_color
    color = lambda text, color='magenta': str(getattr(fb_color, color)(text))
except ImportError as e:
    color = lambda text, color='magenta': str(text)
    print("Exception raised trying to import fabulous!")
    print(e, end='\n'*2)

from actions import Action
from utils import (
    read_action_from_keyboard,
    init_memory
)

from model import(
    DQN,
    Q_table
)

@dataclass
class Player:
    name: str
    money: float=0
    action_type: str='all_money_to_field'
    method_type: str='random_field_and_investment'
    id: t.Optional[int]=None
    history: t.List[Action]=field(default_factory=list)
    cash_history: t.List[float]=field(default_factory=list)
    wins: int=0
    top3: int=0
    leading_role: int=0
    domination_rounds: t.List[int]=field(default_factory=list)
    memory: t.List[int]=field(default_factory=init_memory)
    model_name: str=''

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)
    
    def get_history(self,) -> t.List[Action]:
        return self.history
    
    def get_last_action(self) -> Action:
        return self.history[-1]
    
    def action(self, num_options, money_available, states, epsilon, n_games, decay, total_state, model) -> Action:
        # print(self.method_type)
        # print(self.action_type)
        if self.method_type == 'from_keyboard':
            print(f"Action for player `{color(self.name, color='magenta')}` (player_id: {self.id}):")
            action = read_action_from_keyboard(self.action_type)
        elif self.method_type == 'random':
            #field_num = random.choices([i for i in range(1, num_options)])[0]
            field_num = random.choice([1, 2, 3, 4, 5, 6])
            if self.action_type == 'random_field':
                action = Action(field_id=int(field_num), money_invested=float(inf))
            elif self.action_type == 'random_field_and_investment':
                investment = round(random.uniform(0.0, money_available), 1)
                action = Action(field_id=int(field_num), money_invested=investment) 
        elif self.method_type == 'custom_all_in':
            classes = ['sber_lover', 'lottery_man', 'manufacturer', 'oil_lover', 'gambler', 'cooperator']
            if self.action_type in classes:
                choices = [1, 2, 3, 4, 5, 6]
                if self.action_type == 'sber_lover':
                    field_num = random.choices(choices, weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])[0]
                if self.action_type == 'lottery_man':
                    field_num = random.choices(choices, weights=[0.1, 0.5, 0.1, 0.1, 0.1, 0.1])[0]
                elif self.action_type == 'manufacturer':
                    field_num = random.choices(choices, weights=[0.1, 0.1, 0.5, 0.1, 0.1, 0.1])[0]
                elif self.action_type == 'oil_lover':
                    field_num = random.choices(choices, weights=[0.1, 0.1, 0.1, 0.5, 0.1, 0.1])[0]
                elif self.action_type == 'gambler':
                    field_num = random.choices(choices, weights=[0.1, 0.1, 0.1, 0.1, 0.5, 0.1])[0]
                elif self.action_type == 'cooperator':
                    field_num = random.choices(choices, weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])[0]
                action = Action(field_id=int(field_num), money_invested=float(inf))
            else:
                if self.action_type == 'coop_based':
                    # print(states)
                    if len(list(states.keys())) == 1:
                        action = Action(field_id=5, money_invested=float(inf))
                    else:
                        key = list(states.keys())[-2]
                        state = states[key]
                        # print(state)
                        if state['Field 4']['Number of players'] == 0: 
                            action = Action(field_id=4, money_invested=float(inf))
                        elif 2 <= state['Field 6']['Number of players'] <= 3:
                            action = Action(field_id=6, money_invested=float(inf))
                        elif state['Field 3']['Number of players'] <= 2:
                            action = Action(field_id=3, money_invested=float(inf))
                        else:
                            field_num = random.choice([1, 2, 5])
                            action = Action(field_id=field_num, money_invested=float(inf))
                elif self.action_type == 'memory_based':
                    threshold = random.random()
                    if threshold < 0.3:
                        field_num = random.choices([1, 2, 3, 4, 5, 6])[0]
                    else:
                        field_num = self.memory.index(max(self.memory))+1
                    action = Action(field_id=field_num, money_invested=float(inf))
                elif self.action_type == 'DQN_learning':
                    random_num = random.random()
                    if random_num < max(epsilon - n_games / decay, 0.1):
                        field_num = random.choice([i for i in range(1, num_options)])
                    else:
                        cur_state = torch.tensor(total_state, dtype=torch.float)
                        prediction = model(cur_state)
                        field_num = torch.argmax(prediction).item()+1
                        # action_coord = torch.argmax(prediction).item()
                        # action[action_coord] = 1
                    action = Action(field_id=field_num, money_invested=float(inf))
                elif self.action_type == 'DQN':
                    cur_state = torch.tensor(total_state, dtype=torch.float)
                    if self.model_name == '':
                        prediction = model(cur_state)
                        field_num = torch.argmax(prediction).item()+1
                    else:
                        add_model = DQN()
                        add_model.load(file_name=self.model_name)
                        prediction = add_model(cur_state)
                        field_num = torch.argmax(prediction).item()+1
                    action = Action(field_id=field_num, money_invested=float(inf))        
                elif self.action_type == 'Q_table_learning':
                    random_num = random.random()
                    if random_num < max(epsilon - n_games / decay, 0.1):
                        field_num = random.choice([i for i in range(1, num_options)])
                    else:
                        cur_state = total_state
                        prediction = model.get_pred(cur_state)
                        field_num=prediction
                    action = Action(field_id=field_num, money_invested=float(inf))
                elif self.action_type == 'Q_table':
                    cur_state = total_state
                    if self.model_name == '':
                        prediction = model.get_pred(cur_state)
                        field_num=prediction
                    else:
                        add_model = Q_table(num_states=11000, num_actions=6)
                        add_model.load(file_name=self.model_name)
                        prediction = add_model.get_pred(cur_state)
                        field_num=prediction
                    action = Action(field_id=field_num, money_invested=float(inf))
        # elif self.method_type == 'custom_random_investment':
        #     classes = ['sber_lover', 'loser', 'manufacturer', 'oil_lover', 'gambler', 'cooperator']
        #     field_num = classes.index(self.action_type) + 1
        #     investment = round(random.uniform(0.0, money_available), 1)
        #     action = Action(field_id=int(field_num), money_invested=investment) 
        action.money_invested = min(self.money, action.money_invested)
        self.history.append(action)
        self.money -= action.money_invested
        return action