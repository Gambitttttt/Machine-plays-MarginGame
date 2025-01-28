def basic_metric(dict, id) -> int:
    for player_id, player in dict.items():
        if player_id == id:
            return player.money
        
def discounted_metric(dict, id, discount) -> int:
    for player_id, player in dict.items():
        if player_id == id:
            cash_history = player.cash_history
    cash_history.reverse()
    for i in range(len(cash_history)):
        cash_history[i] = cash_history[i] * (discount ** i)
    return round(sum(cash_history), 3)