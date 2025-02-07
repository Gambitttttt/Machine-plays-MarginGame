import typing as t 
import random
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
from utils import read_action_from_keyboard


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

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)
    
    def get_history(self,) -> t.List[Action]:
        return self.history
    
    def get_last_action(self) -> Action:
        return self.history[-1]
    
    def action(self, num_options, money_available) -> Action:
        if self.method_type == 'from_keyboard':
            print(f"Action for player `{color(self.name, color='magenta')}` (player_id: {self.id}):")
            action = read_action_from_keyboard(self.action_type)
        elif self.method_type == 'random':
            field_num = random.choices([i for i in range(1, num_options)])[0]
            if self.action_type == 'random_field':
                action = Action(field_id=int(field_num), money_invested=float(inf))
            elif self.action_type == 'random_field_and_investment':
                investment = round(random.uniform(0.0, money_available), 1)
                action = Action(field_id=int(field_num), money_invested=investment) 
        elif self.method_type == 'custom_all_in':
            classes = ['sber_lover', 'loser', 'manufacturer', 'oil_lover', 'gambler', 'cooperator']
            field_num = classes.index(self.action_type) + 1
            action = Action(field_id=int(field_num), money_invested=float(inf))
        elif self.method_type == 'custom_random_investment':
            classes = ['sber_lover', 'loser', 'manufacturer', 'oil_lover', 'gambler', 'cooperator']
            field_num = classes.index(self.action_type) + 1
            investment = round(random.uniform(0.0, money_available), 1)
            action = Action(field_id=int(field_num), money_invested=investment)

        action.money_invested = min(self.money, action.money_invested)
        self.history.append(action)
        self.money -= action.money_invested
        return action