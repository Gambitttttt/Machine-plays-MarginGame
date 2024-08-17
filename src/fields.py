import typing as t 
import random 
from dataclasses import dataclass
from dataclasses import field

from src.players import Player, Action

PlayersActions = t.Dict[int, Action]
PlayersRevenues = t.Dict[int, float] # player_id, money

@dataclass
class Field:
    id: int 
    name: str='tmp_name'
    statistics: t.Dict[str, t.Any]=field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)
    
    def return_revenues(
        self, 
        players_actions: PlayersActions
    ) -> PlayersRevenues:
        ...
        
@dataclass
class SberBank(Field):
    
    name: str='SberBank'
    interest_rate: float=0.1
    
    def return_revenues(
        self, 
        players_actions: PlayersActions
    ) -> PlayersRevenues:
        return {
            player_id: action.money_invested * (1+self.interest_rate)
            for player_id, action in players_actions.items()
            if action.field_id == self.id 
        }
        
@dataclass
class CryptoStartup(Field):
    
    name: str='CryptoStartup'
    success_probability: float=0.16
    multiplier: float=3.5
    
    def return_revenues(
        self, 
        players_actions: PlayersActions
    ) -> PlayersRevenues:
        return {
            player_id: action.money_invested * (self.multiplier if random.choices([0, 1], [1-self.success_probability, self.success_probability])[0] == 1 else 0)
            for player_id, action in players_actions.items()
            if action.field_id == self.id 
        }
    