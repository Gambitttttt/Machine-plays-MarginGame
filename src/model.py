import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import os
import random

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        # self.fc2 = nn.Linear(169, 169)
        self.fc3 = nn.Linear(64, 6)

    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, file_name):
        model_folder_path = './src'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        model_folder_path = './src'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        
class trainer():
    def __init__(self, model, lr, gamma, target_model):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.step = 0
        self.loss_log = []
        self.grad_log = []
        self.grad_stats_log = []


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.int64)
        reward = torch.tensor(reward, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        done = torch.tensor(done, dtype = torch.float)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)
        pred = self.model(state)
        # actual_actions = action
        # actual_actions = torch.argmax(action, dim = 1).unsqueeze(1)
        # print(f'PRED: {pred}')
        # print('-'*100)
        # print(f'STATE SHAPE: {state.shape}')
        # print("State sample:", state[0])
        # print("Pred sample:", pred[0])
        # print('-'*100)
        action = torch.unsqueeze(action, 1)
        # print(f'ACTIONS: {action}')
        # print(f'REWARD: {reward.unsqueeze(1)}')
        # print(f'ACTIONS_INDICES: {actual_actions}')
        Q_pred = pred.gather(1, action)

        with torch.no_grad():
            # print(f'Target_model from next_state: {self.target_model(next_state)}')
            
            # next_q_values = self.target_model(next_state).max(1, keepdim=True)[0]              # Без DDQN
            next_actions = self.model(next_state).argmax(dim=1, keepdim=True)  # Основная сеть выбирает действие + затем DDQN
            next_q_values = self.target_model(next_state).gather(1, next_actions)     # DDQN

            # print(f'Наилучшие действия: {next_q_values}')
            # print(f'REWARD: {reward.unsqueeze(1)}')
            # print(f'1 - DONE: {1-done}')
            target_q_values = torch.unsqueeze(reward, 1) + torch.unsqueeze((1 - done),1) * self.gamma * next_q_values
            # print(f'TARGET_Q_VALS: {target_q_values}')

        # print("\n=== TRAIN STEP ===")
        # print(f"State sample: {state[0].numpy()}")
        # print(f"Predicted Q-values: {pred[0].detach().numpy()}")
        # print(f"Chosen action: {action[0].item()}, Q_pred: {Q_pred[0].item()}")
        # print(f"Target Q-value: {target_q_values[0].item()}")
        # print(f"Loss: {self.criterion(Q_pred, target_q_values).item()}")

        self.optimizer.zero_grad()
        loss = self.criterion(target_q_values, Q_pred)
        loss.backward()
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad mean: {param.grad.mean().item():.6f}, std: {param.grad.std().item():.6f}")

        # === Градиент: глобальная норма ===
        
        # total_norm = 0
        # grads = []
        # for name, param in self.model.named_parameters():          # ДЛЯ L2
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2)
        #         grads.append(param.grad.view(-1))
        #         total_norm += grad_norm.item() ** 2
        # total_norm = total_norm ** 0.5

        # 1. Считаем суммарную L1-норму градиентов
        total_norm = 0.0
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.abs().sum()
                grads.append(p.grad.view(-1))
                total_norm += param_norm.item()

        all_grads = torch.cat(grads)
        grad_min = all_grads.min().item()
        grad_max = all_grads.max().item()
        grad_mean = all_grads.mean().item()
        if self.step % 10 == 0:
        # Логируем
            self.loss_log.append(loss.item())
            self.grad_log.append(total_norm)
            self.grad_stats_log.append((grad_min, grad_mean, grad_max))

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)     # Для MSE
        max_norm = 1.0  # максимальная допустимая L1-норма градиентов

        # # 2. Вычисляем коэффициент обрезки
        clip_coef = max_norm / (total_norm + 1e-6)  # добавим eps во избежание деления на 0

        # # 3. Применяем обрезку, если коэффициент < 1
        if clip_coef < 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        self.optimizer.step()

        tau = 0.01
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        self.step += 1
        
class Q_table():
    def __init__(self, num_states, num_actions):
        self.table = np.zeros((num_states, num_actions))
    
    def transform_state(self, state, turns_num=15, up_treshold_1=8, up_treshold_2=8, up_treshold_3=8):
        state_transformed=state.copy()
        state_transformed[0] = state_transformed[0] * turns_num
        state_transformed[1] = state_transformed[1] * up_treshold_1
        state_transformed[2] = state_transformed[2] * up_treshold_2
        state_transformed[3] = state_transformed[3] * up_treshold_3
        integer=state_transformed[0]*up_treshold_1*up_treshold_2*up_treshold_3 + state_transformed[1]*up_treshold_2*up_treshold_3 + state_transformed[2]*up_treshold_3 + state_transformed[3]
        return int(integer)
    
    def get_pred(self, state):
        transformed_state = self.transform_state(state)
        # weights = self.table[transformed_state]
        # print(weights)
        # weights = weights / sum(weights) if sum(weights) != 0 else [1/6] * 6
        return np.argmax(self.table[transformed_state])+1 # Выбор действия с max Q-val
        # return random.choices([1, 2, 3, 4, 5, 6], weights=weights)[0] # Выбор действия логит Q-val
    
    def save(self, name):
        model_folder_path = './src'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, name)
        np.save(file_name, self.table)

    def load(self, file_name):
        model_folder_path = './src'
        file_name = os.path.join(model_folder_path, file_name)
        self.table = np.load(file_name)
    
class Q_table_trainer():
    def __init__(self, model, lr, gamma):
        self.model=model
        self.lr=lr
        self.gamma=gamma

    def train_step(self, state, action, reward, next_state, done):
        state_transformed=self.model.transform_state(state)
        next_state_transformed=self.model.transform_state(next_state)
        action -= 1
        Q_table = self.model.table
        if not done:
            Q_table[state_transformed][action] += self.lr*(reward + self.gamma*np.max(Q_table[next_state_transformed]) - Q_table[state_transformed][action])
        else:
            Q_table[state_transformed][action] = reward