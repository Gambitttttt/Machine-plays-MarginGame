from src.fields import (
    # Field,
    SberBank,
    CryptoStartup,
    Manufactory
)
from src.players import Player

# GAME_CONFIG = {
#     "players": [
#         {"name": "Anton", "money": 10, "id": 1},
#         {"name": "Tata", "money": 10, "id": 2},
#         {"name": "Dima", "money": 10, "id": 3},
#         {"name": "Yarik", "money": 10, "id": 4}
#     ],
#     "fields": [
#         SberBank(id=1, interest_rate=0.1),
#         CryptoStartup(id=2, success_probability=0.16, multiplier=3.5),
#         Manufactory(id=3, total_players_threshold=2, high_multiplier=2.1, low_multiplayer=0.2),
#     ]
# }

GAME_CONFIG = {
    "n_iterations": 2,
    "players": {
        1: Player.from_dict({"name": "Anton", "money": 10, "id": 1}),
        2: Player.from_dict({"name": "Tata", "money": 10, "id": 2}),
        3: Player.from_dict({"name": "Dima", "money": 10, "id": 3}),
        4: Player.from_dict({"name": "Yarik", "money": 10, "id": 4}),
    },
    "fields": {
        1: SberBank(id=1, interest_rate=0.1),
        2: CryptoStartup(id=2, success_probability=0.16, multiplier=3.5),
        3: Manufactory(id=3, total_players_threshold=2, high_multiplier=2.1, low_multiplayer=0.2),
    }
}