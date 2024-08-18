import typing as t 
from dataclasses import dataclass

@dataclass
class Action:
    field_id: int
    money_invested: float 
    
    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)