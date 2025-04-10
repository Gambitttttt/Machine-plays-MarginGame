import json
import matplotlib.pyplot as plt

def overall_vis():
    file_path='stats.json'
    with open(file_path, "r", encoding="utf-8") as file:
        data=json.load(file)
    path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/Overall_vis/'
    for type in ['manufacturer','oil_lover','gambler','cooperator','coop_based','memory_based', 'Q_table', 'DQN', 'Main']:
        for metric in ['wins','top3','domination_rounds','basic metric q1','basic metric q2','basic metric q3','discounted metric q1','discounted metric q2','discounted metric q3']:
            plt.clf()
            plt.hist(data[type][metric], bins=50, density=True) #density=True, bins=50?
            plt.title(type + ':' + metric)
            name=path+type+'_'+metric+'.png'
            plt.savefig(name)

def comparison_vis():
    file_path='stats.json'
    with open(file_path, "r", encoding="utf-8") as file:
        data=json.load(file)
    path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/Comparison_vis/'
    for metric in ['wins','top3','domination_rounds','basic metric q1','basic metric q2','basic metric q3','discounted metric q1','discounted metric q2','discounted metric q3']:
        plt.clf()
        for type in data.keys():
            plt.hist(data[type][metric], bins=50, density=True, label=type, alpha=0.5) #density=True, bins=50?
        plt.legend(list(data.keys()))
        plt.title('All:' + metric)
        name=path+'Everybody_'+metric+'.png'
        plt.savefig(name)

def duo_comparison_vis():
    file_path='stats.json'
    with open(file_path, "r", encoding="utf-8") as file:
        data=json.load(file)
    path = 'C:/Users/WS user/MarginGame/MarginGame2/MarginGame/Graphs/Duo_comparison_vis/'
    for metric in ['wins','top3','domination_rounds','basic metric q1','basic metric q2','basic metric q3','discounted metric q1','discounted metric q2','discounted metric q3']:
        for type in data.keys():
            if type != 'Main':
                plt.clf()
                plt.hist(data[type][metric], bins=50, density=True, label=type, alpha=0.5) #density=True, bins=50?
                plt.hist(data['Main'][metric], bins=50, density=True, label=type, alpha=0.5) #density=True, bins=50?
                plt.legend([type, 'Main'])
                plt.title(f'Main vs {type}: {metric}')
                name=path+f'Main vs {type}_{metric}.png'
                plt.savefig(name)

if __name__ == '__main__':
    #overall_vis() #в целом увидеть картину с отдельными распределениями  для каждого типа и каждой хар-ки
    #comparison_vis() # сравнение распределений хар-ки сразу между всеми классами - мешанина
    duo_comparison_vis() # попарное сравнение хар-к между главным и побочными классами