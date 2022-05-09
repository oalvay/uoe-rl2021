import gym
import numpy as np
from collections import namedtuple
from abc import ABC
env = gym.make('Taxi-v3')
env.reset()
import os

Step = namedtuple("Step", ["act", "obs", 'reward'])

gamma = 0.95
epsilon = 0.9

class MDP(ABC):
    def __init__(self, env, gamma, max_deduct):
        self.obs_n = env.observation_space.n
        self.act_n = env.action_space.n
        self.gamma = gamma
        self.Q = np.zeros((self.obs_n, self.act_n))
        self.epsilon = 1
        self.max_deduct = max_deduct

    def act(self, obs):
        throw_dice = np.random.uniform()
        if throw_dice < self.epsilon:
            return np.random.choice(self.act_n)
        else:
            act_vals = self.Q[int(obs)]
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max(act_vals)]
            return np.random.choice(max_acts)

    def update_epsilon(self):
        try:
            self.update_count += 1
        except:
            self.update_count = 1
        self.epsilon = max(0, 0.8 * (self.max_deduct - self.update_count) / self.max_deduct + 0)

    def generate_episode(self, max_step, act_greedy = False):
        if act_greedy == True:
            self.epsilon = 0
        obs = env.reset()
        episode = []
        for _ in range(max_step):
        # while True:
            act = self.act(obs)
            new_obs, reward, done, _ = env.step(act)
            episode.append(Step(act=act, obs=obs, reward=reward))
            if done:
                break
            obs = new_obs
        return episode, done

    def eval_policy(self):
        avg_reward = 0
        dones = 0
        for i in range(500):
            episode, done = self.generate_episode(100, act_greedy=True)
            G = sum([step.reward for step in episode])
            avg_reward += G / 500
            dones += G > 0
        return avg_reward, dones

    def train(self, k):
        for i in range(k):
            self.update_epsilon()
            self.train_once()
            if (i + 1) % 10000 == 0:
                print(self.update_count)
                print(f'{i}, epsilon: {self.epsilon}')
                print(self.eval_policy())
                
class Sarsa_Agent(MDP):
    def __init__(self, *args):
        super().__init__(*args)
        self.alpha = 0.5

    def train_once(self):
        obs = env.reset()
        act = self.act(obs)
        done = False
        while not done:
            new_obs, reward, done, _ = env.step(act)
            next_act = self.act(new_obs)
            self.Q[obs, act] += self.alpha * (reward + self.gamma * self.Q[new_obs, next_act] -\
                                             self.Q[obs, act])
            obs, act = new_obs, next_act
            
class SarsaLambda_Agent(MDP):
    def __init__(self, *args, alpha, lambda_):
        super().__init__(*args)
        self.alpha = alpha
        self.lambda_ = lambda_
    
    def train_once(self):
        obs = env.reset()
        act = self.act(obs)
        done = False
        E = np.zeros((self.obs_n, self.act_n)) # eligibility trace
        while not done:
            new_obs, reward, done, _ = env.step(act)
            next_act = self.act(new_obs)
            delta = reward + self.gamma * self.Q[new_obs, next_act] - self.Q[obs, act]
            E[obs, act] += 1
            self.Q += self.alpha * delta * E
            E = self.gamma * self.lambda_ * E
            obs, act = new_obs, next_act
            
class QLearning_Agent(MDP):
    def __init__(self, *args):
        super().__init__(*args)
        self.alpha = 0.5

    def act_greedy(self):
        act_vals = self.Q[int(obs)]
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max(act_vals)]
        return np.random.choice(max_acts)

    def train_once(self):
        obs = env.reset()
        act = self.act(obs)
        done = False
        while not done:
            new_obs, reward, done, _ = env.step(act)
            next_act = self.act_greedy(new_obs)
            self.Q[obs, act] += self.alpha * (reward + self.gamma * self.Q[new_obs, next_act] -\
                                             self.Q[obs, act])
            obs, act = new_obs, next_act
       
agt = SarsaLambda_Agent(env, 0.99, 50000, alpha = 0.5, lambda_ = 0.9)
agt.train(100000)
