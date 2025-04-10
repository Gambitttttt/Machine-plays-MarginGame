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

def bar_stats_reduced(exp_type, names, stats, ids, stats_name, n_metrics=3):
    players = names
    for i in range(n_metrics):
        stats_reduced = []
        for id in ids:
            stats_reduced.append(stats[i][id-1])
        fig, ax = plt.subplots()
        ax.bar(players, stats_reduced)
        ax.set_ylabel(stats_name[i])
        title = exp_type + ' ' + stats_name[i]
        ax.set_title(title)
        path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/'
        plt.savefig(path + title + '.png')

def agents_comparison_reduced(exp_type, names, ids, dict_stats, xlim, metric_types, n_distributions=2):
    for j in range(n_distributions):
        fig, ax = plt.subplots(1, len(ids))
        for k in range(len(ids)):
            ax[k].hist(np.array(dict_stats[j][str(ids[k])]), bins = 20, range=(0,xlim), density=True)
            ax[k].set_title(names[k] + ' ' + metric_types[j])
        path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/'
        title = exp_type + ' ' + metric_types[j] + '.png'
        plt.savefig(path+title)

def agents_comparison_uni(exp_type, names, ids, dict_stats, xlim, metric_types, n_distributions=2):
    for j in range(n_distributions):
        fig, ax = plt.subplots()
        for k in range(len(ids)):
            plt.hist(np.array(dict_stats[j][str(ids[k])]), bins = 50, range=(0,xlim), density=True, label = names[k], alpha = 0.5)
            #sns.distplot(np.array(dict_stats[j][str(ids[k])]), bins = 25, range=(0,xlim), density=True, label = names[k], hist_kws={"alpha": 0.5})
        plt.legend(names)
        path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/'
        title = exp_type + ' ' + metric_types[j] + '.png'
        plt.savefig(path+title)

def full_vis(exp_type, names, ids, stats, dict_stats, xlim, stats_name, metric_types, 
             bar_func = bar_stats_reduced, compare_func = agents_comparison_uni):
    bar_func(exp_type, names, stats, ids, stats_name, n_metrics=3)
    compare_func(exp_type, names, ids, dict_stats, xlim, metric_types, n_distributions=2)