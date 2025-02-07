import matplotlib.pyplot as plt
import numpy as np

def one_agent(dict_basic_stats, dict_discounted_stats, player_id):
    plt.subplot(3,2,1)
    plt.hist(np.array(dict_basic_stats[str(player_id)]), bins = 50)
    plt.title('Basic metric', fontdict={'size': 8})

    plt.subplot(3,2,3)
    plt.hist(np.array(dict_basic_stats[str(player_id)]), bins = 50, range=(0, 1000))
    plt.title('Basic metric with xlim 1000', fontdict={'size': 8})

    plt.subplot(3,2,5)
    plt.hist(np.array(dict_basic_stats[str(player_id)]), bins = 50, range=(0, 500))
    plt.title('Basic metric with xlim 500', fontdict={'size': 8})

    plt.subplot(3,2,2)
    plt.hist(np.array(dict_discounted_stats[str(player_id)]), bins = 50)
    plt.title('Discounted metric', fontdict={'size': 8})

    plt.subplot(3,2,4)
    plt.hist(np.array(dict_discounted_stats[str(player_id)]), bins = 50, range=(0, 2000))
    plt.title('Discounted metric with xlim 2000', fontdict={'size': 8})

    plt.subplot(3,2,6)
    plt.hist(np.array(dict_discounted_stats[str(player_id)]), bins = 50, range=(0, 1000))
    plt.title('Discounted metric with xlim 1000', fontdict={'size': 8})
            
    plt.show()

def agents_comparison(dict_stats, xlim, n_players, metric_type):
    for i in range(1, n_players+1):
        plt.subplot(3,3,i)
        plt.hist(np.array(dict_stats[str(i)]), bins = 50, range=(0,xlim))
        plt.title(f'{metric_type} metric for player {i}', fontdict={'size': 8})
    plt.show()

def bar_stats(n_players, stats, stats_name):
    fig, ax = plt.subplots()
    players = [f'Player {i}' for i in range(1, n_players+1)]
    stats = stats
    ax.bar(players, stats)
    ax.set_ylabel(stats_name)
    plt.show()