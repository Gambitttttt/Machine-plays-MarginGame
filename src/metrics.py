def basic_metric(dict, id) -> int:
    for player_id, player in dict.items():
        if player_id == id:
            return player.money
        
def discounted_metric(dict, id) -> int:
    for player_id, player in dict.items():
        if player_id == id:
            return player.cash_history