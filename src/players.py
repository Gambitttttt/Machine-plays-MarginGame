import typing as t 
from dataclasses import dataclass
from dataclasses import field 

from src.fields import Field

@dataclass
class Action:
    field_id: int
    money_invested: float 
    
    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)
        
    

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
    