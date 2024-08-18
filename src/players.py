import typing as t 
from dataclasses import dataclass
from dataclasses import field 

from src.actions import Action
from src.utils import read_action_from_keyboard


@dataclass
class Player:
    name: str
    money: float=0
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
        print(f"Action for player_id: {self.id}")
        action = read_action_from_keyboard()
        self.history.append(action)
        return action
    