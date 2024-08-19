import typing as t 
from dataclasses import dataclass
from dataclasses import field 

try:
    from fabulous import color as fb_color
    color = lambda text, color='magenta': str(getattr(fb_color, color)(text))
except ImportError as e:
    color = lambda text, color='magenta': str(text)
    print("Exception raised trying to import fabulous!")
    print(e, end='\n'*2)

from src.actions import Action
from src.utils import read_action_from_keyboard


@dataclass
class Player:
    name: str
    money: float=0
    action_type: str='all_money_to_field'
    id: t.Optional[int]=None
    history: t.List[Action]=field(default_factory=list)
    
    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)
    
    def get_history(self,) -> t.List[Action]:
        return self.history
    
    def get_last_action(self) -> Action:
        return self.history[-1]
    
    def action(self) -> Action:
        print(f"Action for player `{color(self.name, color='magenta')}` (player_id: {self.id}):")
        action = read_action_from_keyboard(self.action_type)
        action.money_invested = min(self.money, action.money_invested)
        self.history.append(action)
        return action
    