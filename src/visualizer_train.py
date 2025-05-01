import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def vis_loss_grad(trainer, n_games):
    # steps = range(len(trainer.loss_log))
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title('Loss over time')
    plt.xlabel('Training_step')
    plt.ylabel('Loss')
    plt.plot(trainer.grad_log, label='Gradient L1-Norm', color='green')

    plt.subplot(1, 2, 2)
    plt.title('Gradient Statistics')
    plt.xlabel('Training_step')
    plt.ylabel('Gradient Value')
    grad_min = [x[0] for x in trainer.grad_stats_log]
    grad_mean = [x[1] for x in trainer.grad_stats_log]
    grad_max = [x[2] for x in trainer.grad_stats_log]
    plt.plot(grad_mean, '--', label='Grad Mean', color='orange')
    plt.plot(grad_min, ':', label='Grad Min', color='red', alpha=0.5)
    plt.plot(grad_max, ':', label='Grad Max', color='purple', alpha=0.5)
    plt.grid(True)
    plt.legend()

    # plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(.1)
    plt.savefig(f'{n_games}.png')

def vis_loss(trainer):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    steps = range(len(trainer.loss_log))
    plt.title('Loss over time')
    plt.xlabel('Training_step')
    plt.ylabel('Loss')
    plt.plot(steps, trainer.grad_log, label='Gradient L1-Norm', color='green')
    plt.show(block=False)
    plt.pause(.1)

def vis_grad(trainer):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    steps = range(len(trainer.loss_log))
    plt.title('Gradient Statistics')
    plt.xlabel('Training_step')
    plt.ylabel('Gradient Value')
    grad_min = [x[0] for x in trainer.grad_stats_log]
    grad_mean = [x[1] for x in trainer.grad_stats_log]
    grad_max = [x[2] for x in trainer.grad_stats_log]
    plt.plot(steps, grad_mean, '--', label='Grad Mean', color='orange')
    plt.plot(steps, grad_min, ':', label='Grad Min', color='red', alpha=0.5)
    plt.plot(steps, grad_max, ':', label='Grad Max', color='purple', alpha=0.5)
    plt.show(block=False)
    plt.pause(.1)    