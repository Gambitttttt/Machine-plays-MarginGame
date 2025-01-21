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

from actions import Action
from constants import (
    Players,
    PlayersActions,
    PlayersRevenues,
    FIELD_ID
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
        players: Players,
    ) -> PlayersRevenues:
        ...
    
    def return_rate(
            self
    ) -> float:
        ...
    
Fields = t.Dict[FIELD_ID, Field]

@dataclass
class SberBank(Field):
    
    name: str='SberBank'
    interest_rate: float=0.1
    outcome: float=0.0 

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
        players: Players,
    ) -> PlayersRevenues:
        players_revenues = {}
        self.outcome = self.interest_rate
        for player_id, player in players.items():
            action = player.get_last_action()
            if action.field_id == self.id:
                players_revenues[player_id] = round(action.money_invested * (1+self.interest_rate), self.money_round_digits)
        return players_revenues
    
    def return_rate(
        self
    ) -> float:
        return self.outcome
        
@dataclass
class CryptoStartup(Field):
    
    name: str='CryptoStartup'
    success_probability: float=0.16
    multiplier: float=3.5
    oucome: float=0.0
    
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
        players: Players,
    ) -> PlayersRevenues:
        players_revenues = {}
        multiplier = self.multiplier if random.choices([0, 1], [1-self.success_probability, self.success_probability])[0] == 1 else 0
        self.outcome = multiplier
        for player_id, player in players.items():
            action = player.get_last_action()
            if action.field_id == self.id:
                players_revenues[player_id] = round(action.money_invested * multiplier, self.money_round_digits)
        return players_revenues

    def return_rate(
        self
    ) -> float:
        return self.outcome

@dataclass
class Manufactory(Field):
    
    name: str='Manufactory'
    total_players_threshold: int=2
    high_multiplier: float=2.1
    low_multiplayer: float=0.1
    outcome: float=0.0
    
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
        players: Players,
    ) -> PlayersRevenues:
        total_players = len([
            player_id
            for player_id, player in players.items()
            if player.get_last_action().field_id == self.id
        ])
        resulting_multiplier = (
            self.high_multiplier if total_players <= self.total_players_threshold
            else self.low_multiplayer
        )
        self.outcome = resulting_multiplier
        players_revenues = {}
        for player_id, player in players.items():
            action = player.get_last_action()
            if action.field_id == self.id:
                players_revenues[player_id] = round(action.money_invested * resulting_multiplier, self.money_round_digits)
        return players_revenues
    
    def return_rate(
        self
    ) -> float:
        return self.outcome

@dataclass
class OilCompany(Field):
    
    name: str='OilCompany'
    intercept: float=4.0
    slope: float=-1.0
    minimum_return_value: float=0.0
    outcome: float=0.0

    @property
    def description(self):
        return (
            f"""
            {self.name}
            {color('The revenue formula:', color='yellow')}
            revenue = max({self.minimum_return_value}, {self.slope} x total_amount_of_investors + {self.intercept}) x invested_money 
            """
        )
    
    def return_revenues(
        self, 
        players: Players,
    ) -> PlayersRevenues:
        total_players = len([
            player_id
            for player_id, player in players.items()
            if player.get_last_action().field_id == self.id
        ])
        resulting_multiplier = max(0.0, self.slope * total_players + self.intercept)
        self.outcome = resulting_multiplier
        players_revenues = {}
        for player_id, player in players.items():
            action = player.get_last_action()
            if action.field_id == self.id:
                players_revenues[player_id] = round(
                    resulting_multiplier * action.money_invested, 
                    self.money_round_digits
                )
        return players_revenues
    
    def return_rate(
        self
    ) -> float:
        return self.outcome

@dataclass
class Profit(Field):

    name: str='Profit'
    lucky_outcome: float=3.0
    unlucky_outcome: float=0.7
    ok_outcome: float=1.5
    lucky_prob: float=0.25
    unlucky_prob: float=0.25
    ok_prob: float=0.5
    outcome: float=0.0

    @property
    def description(self):
        return (
            f"""
            {self.name} is completely dependent on random
            {color('The revenue formula:', color='yellow')}
            revenue = (invested_money x {self.lucky_outcome}) with probability = {self.lucky_prob} or you get 
            (invested_money x {self.unlucky_outcome}) with probability = {self.unlucky_prob} or you get
            (invested_money x {self.ok_outcome}) with probability = {self.ok_prob}
            """     
            )
    
    def return_revenues(
        self, 
        players: Players,
    ) -> PlayersRevenues:
        players_revenues = {}
        resulting_multiplier = random.choices([self.lucky_outcome, self.unlucky_outcome, self.ok_outcome], 
                                 weights = [self.lucky_prob, self.unlucky_prob, self.ok_prob])[0]
        self.outcome = resulting_multiplier
        for player_id, player in players.items():
            action = player.get_last_action()
            if action.field_id == self.id:
                players_revenues[player_id] = round(action.money_invested * resulting_multiplier, self.money_round_digits)
        return players_revenues
    
    def return_rate(
        self
    ) -> float:
        return self.outcome
    
@dataclass
class New_Sector(Field):

    name: str='New Sector'
    threshold_upper: int=10
    threshold_lower: int=5
    high_multiplier: float=2.5
    low_multiplier: float=0.8
    outcome: float=0.0

    @property
    def description(self):
        return (
            f"""
            {self.name} is dependent on the total number of investors 
            {color('The revenue formula:', color='yellow')}
            revenue = (invested_money x {self.high_multiplier}) if the number of investors is 
            between {self.threshold_lower} and {self.threshold_upper} otherwise you get
            (invested_money x {self.low_multiplier})
            """     
            )
    
    def return_revenues(
        self, 
        players: Players,
    ) -> PlayersRevenues:
        total_players = len([
            player_id
            for player_id, player in players.items()
            if player.get_last_action().field_id == self.id
        ])
        resulting_multiplier = (
            self.high_multiplier if self.threshold_lower <= total_players <= self.threshold_upper
            else self.low_multiplier
        )
        self.outcome = resulting_multiplier
        players_revenues = {}
        for player_id, player in players.items():
            action = player.get_last_action()
            if action.field_id == self.id:
                players_revenues[player_id] = round(action.money_invested * resulting_multiplier, self.money_round_digits)
        return players_revenues

    def return_rate(
            self
        ) -> float:
            return self.outcome