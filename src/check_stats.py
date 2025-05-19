import json
import matplotlib.pyplot as plt
import numpy as np

def view_stats():
    rusifikator = {'wins': 'Доля побед игроков', 'top3': 'Доля попаданий в тройку лучших', 'domination_rounds': 'Доля раундов с доминирующим положением',
                   'memory_based': 'Памятливый игрок', 'coop_based': 'Кооперативный игрок', 'cooperator': 'Предприниматель',
                   'gambler': 'Игрок', 'oil_lover': 'Нефтяник', 'manufacturer': 'Промышленник', 'lottery_man': 'Криптоинвестор', 
                   'sber_lover': 'Сберегатель', 'Main': 'Обученный агент', 'Q_table': 'Q-обучение', 'DQN': 'Q-сеть'}
    file_path='stats.json'
    with open(file_path, "r", encoding="utf-8") as file:
        data=json.load(file)
    for metric in ['wins','top3','domination_rounds']:
        plt.clf()
        plt.figure(figsize=(15, 6))
        names = []
        stats = []
        print(f'{metric}:')
        for type in data.keys():
            if not type in ['Z']:     
                names.append(rusifikator[type])
                stats.append(np.mean(data[type][metric]))
                print(f'{type}: в среднем {np.mean(data[type][metric])}')
        plt.barh(np.array(names), np.array(stats))
        plt.title(f'{rusifikator[metric]}, %')
        plt.savefig(f'{rusifikator[metric]}.png')
        print()

if __name__ == '__main__':
    view_stats()