def basic_metric(dict, id) -> float:
    return round(dict[id].money,3)
        
def discounted_metric(dict, id, discount) -> float:
    cash_history = dict[id].cash_history        
    cash_history.reverse()
    for i in range(len(cash_history)):
        cash_history[i] = cash_history[i] * (discount ** i)
    return round(sum(cash_history), 3)

def wins_metric(dict, id) -> int:
    return dict[id].wins

def top3_metric(dict, id) -> int:
    return dict[id].top3

def mean_rounds_domination(dict, id) -> float:
    return round(sum(dict[id].domination_rounds)/len(dict[id].domination_rounds),2)