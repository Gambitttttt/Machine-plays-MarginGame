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
    action_type: str='simple'
    method_type: str='random_field'
    id: t.Optional[int]=None
    history: t.List[Action]=field(default_factory=list)
    cash_history: t.List[float]=field(default_factory=list)
    
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
        elif self.method_type == 'random_field':
            field_num = random.choices([i for i in range(1, num_options)])[0]
            action = Action(field_id=int(field_num), money_invested=float(inf))
        elif self.method_type == 'random_field_and_investment':
            field_num = random.choices([i for i in range(1, num_options)])[0]
            investment = round(random.uniform(0.0, money_available), 1)
            action = Action(field_id=int(field_num), money_invested=investment)
        elif self.method_type == 'risk_averse_agent':
            field_num = 1
            action = Action(field_id=int(field_num), money_invested=float(inf))
        elif self.method_type == 'risk_taking_agent':
            field_num = 5
            action = Action(field_id=int(field_num), money_invested=float(inf))


        action.money_invested = min(self.money, action.money_invested)
        self.history.append(action)
        self.money -= action.money_invested
        return action