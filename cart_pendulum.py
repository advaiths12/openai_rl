import gym
import random
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DeepQ(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, epsilon, lr, discount):
        super(DeepQ, self).__init__()
        self.epsilon = epsilon
        self.lr = lr
        self.num_hidden = num_hidden
        self.num_states = num_states
        self.discount = discount
        self.num_actions = num_actions
        self.fc1 = torch.nn.Linear(self.num_states, num_hidden)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(num_hidden, num_actions)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def select_action(self, state):
        if(random.random() > self.epsilon):
            with torch.no_grad():
                return torch.argmax(self.forward(state)).item()
        else:
            return torch.argmax(torch.randn(self.num_actions)).item()

    def optimize(self, nobs, pobs, reward, action, optimizer):
        td_error = F.smooth_l1_loss(reward + self.discount*torch.max(self.forward(nobs)), self.forward(pobs)[action])
        optimizer.zero_grad()
        td_error.backward()
        optimizer.step()

def q(obs, weights, bias, action=-1):
    if(action == -1):
        return obs.dot(weights) + bias
    else:
        return (obs.dot(weights) + bias)[action]


env = gym.make('CartPole-v0')
env.reset()
max_iter = 500
max_episodes = 10000
epsilon = .10
lr = .01
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
bias = 0
discount = .9
DQN = DeepQ(num_states, num_actions, 16, epsilon, lr, discount)
optimizer = optim.RMSprop(DQN.parameters())
rewards = []
time_steps = []
time_step = 0

def plot_rewards(rewards, ts):
    times = range(0, ts)
    plt.clf()
    plt.plot(times, rewards)
    plt.draw()
    plt.pause(.01)


plt.figure()
for tr in range(max_episodes):
    obs = env.reset()
    sum = 0
    for _ in range(max_iter):
        env.render()
        r = np.random.uniform(0.0, 1.0)
        action = DQN.select_action(torch.tensor(obs).float())
        nobs, reward, done, info = env.step(action) # take a random action
        DQN.optimize(torch.tensor(nobs).float(), torch.tensor(obs).float(), torch.tensor(reward).float(), action, optimizer)
        sum += reward
        if(done):
            break
        obs = nobs
    print("Trial: " + str(tr) + "Total Reward:" + str(sum))
    time_step += 1
    rewards.append(sum)
    plot_rewards(rewards, time_step)

env.close()