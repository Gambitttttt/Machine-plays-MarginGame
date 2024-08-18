import typing as t 
import random 
from dataclasses import dataclass
from dataclasses import field

# from src.players import Player
from src.actions import Action
from src.constants import (
    PlayersActions,
    PlayersRevenues,
    FIELD_ID,
)

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
        
Fields = t.Dict[FIELD_ID, Field]

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
    
@dataclass
class Manufactory(Field):
    
    name: str='Manufactory'
    total_players_threshold: int=2
    high_multiplier: float=2.1
    low_multiplayer: float=0.1
    
    def return_revenues(
        self, 
        players_actions: PlayersActions
    ) -> PlayersRevenues:
        total_players = len([
            player_id
            for player_id, action in players_actions.items()
            if action.field_id == self.id
        ])
        resulting_multiplier = (
            self.high_multiplier if total_players <= self.total_players_threshold
            else self.low_multiplayer
        )
        return {
            player_id: action.money_invested * resulting_multiplier
            for player_id, action in players_actions.items()
            if action.field_id == self.id 
        }