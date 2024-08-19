Margin game

python version: 3.10.13

To play the game:

1) Prepare json config like this:

```json
{
    "n_iterations": 5,
    "read_action_type": "all_money_to_field",
    "players": {
        "1": {"name": "Anton", "money": 10, "id": 1, "action_type": "all_money_to_field"},
        "2": {"name": "Tata", "money": 10, "id": 2, "action_type": "all_money_to_field"},
        "3": {"name": "Dima", "money": 10, "id": 3, "action_type": "all_money_to_field"},
        "4": {"name": "Yarik", "money": 10, "id": 4, "action_type": "all_money_to_field"}
    },
    "fields": {
        "1": "SberBank(id=1, interest_rate=0.1)",
        "2": "CryptoStartup(id=2, success_probability=0.1, multiplier=5.5)",
        "3": "Manufactory(id=3, total_players_threshold=2, high_multiplier=2.1, low_multiplayer=0.2)"
    }
}
```

2) 
a)  Run script with the game:
```bash
# config_path=<path to game config file .json>
cd <this project>; export PYTHONPATH=$PWD; python src/game.py --config_path=./configs/game_config.json
```

b) Alternatively, you can run the game with docker:
```bash
docker run -it --rm -v /path/to/game/config/file.json:/app/configs/game_config.json joitandr/margin_game
```

You'll see such output:

<img width="797" alt="image" src="https://github.com/user-attachments/assets/85c69926-ec7e-4c6d-afd1-6d2112567f1d">
<img width="970" alt="image" src="https://github.com/user-attachments/assets/ffa8d125-8780-4223-b1fc-6f08608d7131">
<img width="753" alt="image" src="https://github.com/user-attachments/assets/97c1aac6-7a69-49a7-aa2a-0bb7ae78ad23">
<img width="709" alt="image" src="https://github.com/user-attachments/assets/6acbbc56-052b-48ce-a663-c3c7dd187857">
<img width="1091" alt="image" src="https://github.com/user-attachments/assets/c91e7ef1-7062-4391-9029-9d3d3d39b296">