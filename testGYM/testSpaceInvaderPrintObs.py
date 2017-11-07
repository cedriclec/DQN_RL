#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:16:42 2017

@author: ced
"""

import gym
env = gym.make('CartPole-v0')
env.reset()
#env.render()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done: 
            print ("Episode finished after {} episodes ".format(t+1))
            break
            