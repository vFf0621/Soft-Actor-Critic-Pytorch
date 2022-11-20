#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:04:16 2022

@author: guanfei1
"""

import gym
import numpy as np
from gym import wrappers
from SAC import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env=gym.make("Walker2d-v4", render_mode="human")

    
    agent = SAC(env)
    ys = []
    eps = 2000
    xs = list(range(eps))
    print("Replay Buffer Initialized")
    for j in range(eps):
        done = False
        episode_reward = 0
        s = env.reset()[0]
        while not done:
            action = agent.act(s)
            s_, r, done, _, _ = env.step(action)
            agent.replay_buffer.append((s, action, s_, r, done))
            s = s_
            episode_reward += r
            agent.train()
        agent.reward_buffer.append(episode_reward)
        mean = np.mean(agent.reward_buffer)
        ys.append(mean)
        
        print("Episode Reward", episode_reward, ", Average Reward", 
              mean)
    plt.plot(xs, ys)
    env = gym.make('Walker2d-v4', render_mode="human")
    while True: 
        done = False
        s = env.reset()[0]
        while not done:
            action = agent.act(s)
            s_, r, done, _, _ = env.step(action)
            s = s_
