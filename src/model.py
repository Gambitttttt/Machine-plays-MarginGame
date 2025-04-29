import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import os

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 169)
        # self.fc2 = nn.Linear(169, 169)
        self.fc3 = nn.Linear(169, 6)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
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
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        done = torch.tensor(done, dtype = torch.float)
        print(state)
        print(f'DONE: {done}')
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)
            # done = (done, )
        print(state)
        pred = self.model(state)
        actual_actions = torch.argmax(action, dim = 1).unsqueeze(1)
        print(f'PRED: {pred}')
        print(f'ACTIONS: {action}')
        print(f'ACTIONS_INDICES: {actual_actions}')
        Q_pred = pred.gather(1, actual_actions)
        print(f'Q_PRED: {Q_pred}')
        # target = pred.clone().detach()
        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         with torch.no_grad():
        #             Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))

            # target[idx][torch.argmax(action[idx]).item()] = Q_new

        with torch.no_grad():
            print(f'Target_model from next_state: {self.target_model(next_state)}')
            next_q_values = self.target_model(next_state).max(1, keepdim=True)[0]
            print(f'Наилучшие действия: {next_q_values}')
            print(f'REWARD: {reward.unsqueeze(1)}')
            print(f'1 - DONE: {1-done}')
            target_q_values = reward.unsqueeze(1) + (1 - done).unsqueeze(1) * self.gamma * next_q_values
            print(f'TARGET_Q_VALS: {target_q_values}')

        # # Q_new = reward

        # # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # # pred.clone()
        # # preds[argmax(action)] = Q_new
        # print(f'Target: {target_q_values}')
        # print(f'Pred: {pred}')
        self.optimizer.zero_grad()
        loss = self.criterion(target_q_values, Q_pred)
        loss.backward()
        self.optimizer.step()

        # state = torch.tensor(state, dtype = torch.float)
        # action = torch.tensor(action, dtype = torch.float)
        # reward = torch.tensor(reward, dtype = torch.float)
        # next_state = torch.tensor(next_state, dtype = torch.float)
        # print(state)
        # if len(state.shape) == 1:
        #     # (1, x)
        #     state = torch.unsqueeze(state, 0)
        #     next_state = torch.unsqueeze(next_state, 0)
        #     action = torch.unsqueeze(action, 0)
        #     reward = torch.unsqueeze(reward, 0)
        #     done = (done, )

        # q_values = self.model(state).gather(1, action)

        # with torch.no_grad():
        #     next_q_values = self.target_model(next_state).max(1, keepdim=True)[0]
        #     target_q_values = reward + (1 - done) * self.gamma * next_q_values

        # self.optimizer.zero_grad()
        # loss = self.criterion(q_values, target_q_values)
        # loss.backward()
        # self.optimizer.step()
        
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
        return np.argmax(self.table[transformed_state])+1
    
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