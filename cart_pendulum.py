import gym
import numpy as np

def q(obs, weights, bias, action=-1):
    if(action == -1):
        return obs.dot(weights) + bias
    else:
        return (obs.dot(weights) + bias)[action]


env = gym.make('CartPole-v0')
env.reset()
max_iter = 500
max_episodes = 10000
epsilon = .15
lr = .01
weights = np.zeros((env.observation_space.shape[0], env.action_space.n))
bias = 0
discount = .9
for tr in range(max_episodes):
    obs = env.reset()
    sum = 0
    for _ in range(max_iter):
        env.render()
        r = np.random.uniform(0.0, 1.0)
        q_p = q(obs, weights, bias)
        if(r < epsilon):
            action = np.argmax(q_p)
        else:
            action = env.action_space.sample()
        nobs, reward, done, info = env.step(action) # take a random action
        weights[:,action] -= lr*(q(obs, weights, bias, action) - (reward + discount*np.argmax(q(nobs, weights, bias))))*obs.T
        bias -= lr*(q(obs, weights, bias, action) - (reward + discount*np.argmax(q(nobs, weights, bias))))
        sum += reward
        if(done):
            break
        obs = nobs
    print("Trial: " + str(tr) + "Total Reward:" + str(sum))
env.close()