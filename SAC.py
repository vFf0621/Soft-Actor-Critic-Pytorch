#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:29:38 2022

@author: guanfei1
"""


import torch
from torch import optim
from torch import nn
from collections import deque
from torch.distributions import Normal
import gym
import random
import numpy as np
class Actor(nn.Module):
    def __init__(self, env, device, hidden = 200, lr = 0.0001):
        super().__init__()
        self.linear1 = nn.Linear(env.observation_space.shape[0], 
                                           hidden + 100)
        self.env=env
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100, hidden)
        self.mu = nn.Linear(hidden, env.action_space.shape[0])
        self.device = device
        self.tanh = nn.Tanh()
        self.optim = optim.Adam(self.parameters(), lr = lr)
        self.action_high = torch.from_numpy(self.env.action_space.high).\
            to(self.device)

        self.sigma = nn.Linear(hidden, env.action_space.shape[0])
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = torch.clamp(sigma, min=10e-8, max=1)

        return mu, sigma        
    
    def sample(self, state, reparametrize):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparametrize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = self.tanh(actions)*self.action_high
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+10e-6)
        log_probs = log_probs.unsqueeze(-1)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs


class Value(nn.Module):
    def __init__(self, env, hidden=200, lr = 0.0001):
        super().__init__()
        self.linear1= nn.Linear(env.observation_space.shape[0], 
                                           hidden + 100)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100, hidden)
                                 
        self.linear3 = nn.Linear(hidden, 1)
        self.optim = optim.Adam(self.parameters(), lr = lr)


    def forward(self, state):
        
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x    


class Critic(nn.Module):
    def __init__(self, env, hidden=200, lr = 0.0001):
        super().__init__()
        self.linear1= nn.Linear(env.observation_space.shape[0], 
                                           hidden + 100)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100 + env.action_space.shape[0],
            hidden)
                                 
        self.linear3 = nn.Linear(hidden, 1)
        self.optim = optim.Adam(self.parameters(), lr = lr)

    def forward(self, state, action):
        if len(action.shape) < len(state.shape):
            action = action.unsqueeze(-1)
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x, action], dim=1))
        x = self.relu(x)
        x = self.linear3(x)

        return x
  
class SAC:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if\
        torch.cuda.is_available() else "cpu")
        self.actor = Actor(env=self.env, device=self.device).to(self.device)
        self.critic1 = Critic(env=self.env).to(self.device)
        self.critic2 = Critic(env=self.env).to(self.device)
        self.actor = Actor(env=self.env, device=self.device).to(self.device)
        self.gamma = 0.99
        self.tau = 0.005

        self.replay_buffer = deque(maxlen=1000000)
        self.loss = torch.nn.MSELoss()
        self.reward_buffer = deque(maxlen=1000)
        self.alpha = 0.2
        
        self.value = Value(env).to(self.device)
        self.target_value = Value(env).to(self.device)
        self.value.load_state_dict(self.target_value.state_dict())
        s = env.reset()[0]
        for i in range(100):
            done = False
            while not done:
                action = self.env.action_space.sample()
                s_, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.append((s, action, s_, reward, done))
                s = s_
    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.to(self.device).float()
        
        return self.actor.sample(state,False)[0].cpu().detach().tolist()
        
    def soft_update(self):
        for value, value_hat in zip(self.value.parameters(),
                                       self.target_value.parameters()):
         value_hat.data.copy_(self.tau * value.data + \
                                 (1 - self.tau) * value_hat.data)


    def sample(self, batch_size = 64):
        t = random.sample(self.replay_buffer, batch_size)
        actions = []
        states = []
        dones = []
        states_ = []
        rewards = []
        for i in t:
            state, action, state_, reward, done  = i
            
            states.append(state)

            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            states_.append(state_)
        
        states = torch.from_numpy(np.array(states)).\
        to(self.device).float()
        actions = torch.from_numpy(np.array(actions)).\
        to(self.device).float()
        rewards = torch.from_numpy(np.array(rewards)).\
            to(self.device).float()
        dones = torch.from_numpy(np.array(dones)).\
            to(self.device)
        states_ = torch.from_numpy(np.array(states_)).to(self.device).float()
        
        return states, actions, states_, rewards, dones
        
    def train(self):
    

        state, action, state_, reward, done = self.sample()

       
        v_ = reward + self.gamma*self.target_value(state_).view(-1)
        v = self.value(state).view(-1)
        v_[done] = 0.0
        act, log_probs = self.actor.sample(state, False)
        q = torch.min(self.critic1(state,act), self.critic2(state, act)).view(-1)
        q_ = q-log_probs.view(-1)*self.alpha
        self.value.optim.zero_grad()
        loss = self.loss(v, q_)
        loss.backward(retain_graph=True)
        self.value.optim.step()
        self.critic1.optim.zero_grad()
        self.critic2.optim.zero_grad()
        c1loss = self.loss( self.critic1(state, action).view(-1),v_)
        c2loss = self.loss( self.critic2(state, action).view(-1), v_)
        c1loss.backward(retain_graph=True)
        c2loss.backward(retain_graph=True)
        self.critic1.optim.step()
        self.critic2.optim.step()
        act, log_probs = self.actor.sample(state, True)
        q = torch.min(self.critic1(state,act), self.critic2(state, act)).view(-1)
        self.actor.optim.zero_grad()
        actor_loss = -torch.mean(q-self.alpha*log_probs.view(-1))
        actor_loss.backward()
        self.actor.optim.step()
        self.soft_update()
        
        
