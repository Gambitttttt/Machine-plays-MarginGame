import typing as t 
import random 
from dataclasses import dataclass
from dataclasses import field

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
    # description: str
    name: str='tmp_name'
    statistics: t.Dict[str, t.Any]=field(default_factory=dict)
    money_round_digits: int=3
    
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
    
    @property
    def description(self):
        return (
            f"""
            {self.name} is a safe straregy to invest money in.
            {color('The revenue formula:', color='yellow')}
            revenue = invested_money x (1 + {self.interest_rate})
            """
        )
        
    
    def return_revenues(
        self, 
        players_actions: PlayersActions
    ) -> PlayersRevenues:
        
        return {
            player_id: round(action.money_invested * (1+self.interest_rate), self.money_round_digits)
            for player_id, action in players_actions.items()
            if action.field_id == self.id 
        }
        
@dataclass
class CryptoStartup(Field):
    
    name: str='CryptoStartup'
    success_probability: float=0.16
    multiplier: float=3.5
    
    @property
    def description(self):
        return (
            f"""
            {self.name} is a risky one!
            {color('The revenue formula:', color='yellow')}
            revenue = (invested_money x {self.multiplier}) with probability = {self.success_probability} or you get 
            0 with probability = {1 - self.success_probability}
            """
        )
    
    def return_revenues(
        self, 
        players_actions: PlayersActions
    ) -> PlayersRevenues:
        return {
            player_id: round(action.money_invested * (self.multiplier if random.choices([0, 1], [1-self.success_probability, self.success_probability])[0] == 1 else 0), self.money_round_digits)
            for player_id, action in players_actions.items()
            if action.field_id == self.id 
        }
    
@dataclass
class Manufactory(Field):
    
    name: str='Manufactory'
    total_players_threshold: int=2
    high_multiplier: float=2.1
    low_multiplayer: float=0.1
    
    @property
    def description(self):
        return (
            f"""
            {self.name} is a good one! Revenue from this field depends on the amount of players, 
            who also invested in it.
            {color('The revenue formula:', color='yellow')}
            revenue = (invested_money x {self.high_multiplier}) if total amount of investors <= {self.total_players_threshold}
            otherwise you get (invested_money x {self.low_multiplayer})
            """
        )
    
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
            player_id: round(action.money_invested * resulting_multiplier, self.money_round_digits)
            for player_id, action in players_actions.items()
            if action.field_id == self.id 
        }