{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import torch  \n",
    "import gym\n",
    "import numpy as np  \n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torch.nn.functional as F\n",
    "from torch.autograd import Variable\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "\n",
    "# hyperparameters\n",
    "hidden_size = 256\n",
    "learning_rate = 3e-4\n",
    "\n",
    "# Constants\n",
    "GAMMA = 0.99\n",
    "num_steps = 300\n",
    "max_episodes = 3000\n",
    "epsilon = 0.10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ActorCritic(nn.Module):\n",
    "    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):\n",
    "        super(ActorCritic, self).__init__()\n",
    "\n",
    "        self.num_actions = num_actions\n",
    "        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)\n",
    "        self.critic_linear2 = nn.Linear(hidden_size, 1)\n",
    "\n",
    "        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)\n",
    "        self.actor_linear2 = nn.Linear(hidden_size, num_actions)\n",
    "    \n",
    "    def forward(self, state):\n",
    "        state = Variable(torch.from_numpy(state).float().unsqueeze(0))\n",
    "        \n",
    "        #g3t the value\n",
    "        value = F.relu(self.critic_linear1(state))\n",
    "        value = self.critic_linear2(value)\n",
    "        \n",
    "        #get the new policy dist\n",
    "        p_d = F.relu(self.actor_linear1(state))\n",
    "        p_d = F.softmax(self.actor_linear2(p_d))\n",
    "        \n",
    "        return value, p_d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "num_inputs = env.observation_space.shape[0]\n",
    "num_outputs = env.action_space.n\n",
    "ac = ActorCritic(num_inputs, num_outputs, hidden_size)\n",
    "def train_actor_agent(env):\n",
    "    \n",
    "    \n",
    "    \n",
    "    ac_opt = optim.Adam(ac.parameters(), lr=learning_rate)\n",
    "    \n",
    "    all_lengths = []\n",
    "    average_lengths = []\n",
    "    all_rewards = []\n",
    "    entropy_term = 0\n",
    "    \n",
    "    for episode in range(max_episodes):\n",
    "        log_probs = []\n",
    "        values = []\n",
    "        rewards = []\n",
    "\n",
    "        state = env.reset()\n",
    "        for steps in range(num_steps):\n",
    "            value, p_d = ac.forward(state)\n",
    "            value = value.detach().numpy()[0,0]\n",
    "            dist = p_d.detach().numpy()\n",
    "            \n",
    "            #epsilon greedy action choice\n",
    "            exp = np.random.uniform(0, 1)\n",
    "            if exp >= epsilon:\n",
    "                action = np.random.choice(num_outputs, p=np.squeeze(dist))\n",
    "            else:\n",
    "                action = np.random.choice(num_outputs)\n",
    "            \n",
    "            log_prob = torch.log(p_d.squeeze(0)[action])\n",
    "            entropy = -np.sum(np.mean(dist) * np.log(dist))\n",
    "            new_state, reward, done, _ = env.step(action)\n",
    "            \n",
    "            rewards.append(reward)\n",
    "            values.append(value)\n",
    "            log_probs.append(log_prob)\n",
    "            entropy_term += entropy\n",
    "            state = new_state\n",
    "            \n",
    "            \n",
    "            if done or steps == num_steps-1:\n",
    "                Qval, _ = ac.forward(new_state)\n",
    "                Qval = Qval.detach().numpy()[0,0]\n",
    "                all_rewards.append(np.sum(rewards))\n",
    "                all_lengths.append(steps)\n",
    "                average_lengths.append(np.mean(all_lengths[-10:]))\n",
    "                if episode % 10 == 0:                    \n",
    "                    sys.stdout.write(\"episode: {}, reward: {}, total length: {}, average length: {} \\n\".format(episode, np.sum(rewards), steps, average_lengths[-1]))\n",
    "                break\n",
    "            \n",
    "        Qvals = np.zeros_like(values)\n",
    "\n",
    "        #iterate Q values\n",
    "        for t in reversed(range(len(rewards))):\n",
    "            Qval = rewards[t] + GAMMA * Qval\n",
    "            Qvals[t] = Qval\n",
    "\n",
    "\n",
    "        values = torch.FloatTensor(values)\n",
    "        Qvals = torch.FloatTensor(Qvals)\n",
    "        log_probs = torch.stack(log_probs)\n",
    "\n",
    "        advantage = Qvals - values\n",
    "        actor_loss = (-log_probs * advantage).mean()\n",
    "        critic_loss = .5*advantage.pow(2).mean()\n",
    "        ac_loss = actor_loss + critic_loss + .001*entropy_term\n",
    "\n",
    "        ac_opt.zero_grad()\n",
    "        ac_loss.backward()\n",
    "        ac_opt.step()\n",
    "    \n",
    "\n",
    "env = gym.make(\"CartPole-v0\")\n",
    "train_actor_agent(env)\n",
    "torch.save(ac.state_dict(), \"./ac.weights\")\n",
    "            \n",
    "            \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "rewards = []\n",
    "ac = ActorCritic(num_inputs, num_outputs, hidden_size)\n",
    "ac.load_state_dict(torch.load(\"./ac_weights\"))\n",
    "ac.eval()\n",
    "for episode in range(max_episodes):\n",
    "    state = env.reset()\n",
    "    for steps in range(num_steps):\n",
    "        value, p_d = ac.forward(state)\n",
    "        value = value.detach().numpy()[0,0]\n",
    "        dist = p_d.detach().numpy()\n",
    "\n",
    "        #epsilon greedy action choice\n",
    "        action = np.random.choice(num_outputs, p=np.squeeze(dist))\n",
    "      \n",
    "\n",
    "        new_state, reward, done, _ = env.step(action)\n",
    "\n",
    "        rewards.append(reward)\n",
    "        state = new_state\n",
    "        env.render()\n",
    "        if done or steps == num_steps-1:\n",
    "            if episode % 10 == 0:                    \n",
    "                sys.stdout.write(\"episode: {}, reward: {}, total length: {}\\n\".format(episode, np.sum(rewards), steps))\n",
    "            break\n",
    "        \n",
    "\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}