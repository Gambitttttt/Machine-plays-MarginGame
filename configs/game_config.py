from src.fields import (
    Field,
    SberBank,
    CryptoStartup
)

GAME_CONFIG = {
    "players": [
        {"name": "Anton", "money": 10, "id": 1},
        {"name": "Tata", "money": 10, "id": 2},
        {"name": "Dima", "money": 10, "id": 3},
        {"name": "Yarik", "money": 10, "id": 4}
    ],
    "fields": [
        SberBank(id=1, interest_rate=0.1),
        CryptoStartup(id=2, success_probability=0.16, multiplier=3.5)
    ]
}