#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:16:42 2017

@author: ced

Get from
https://gym.openai.com/docs/

"""

import gym
from gym import spaces

env = gym.make('SpaceInvaders-v0')
env.reset()
print("Action possible {}".format(env.action_space))

print("Observation space {}".format(env.observation_space))

print(env.observation_space.high)

print(env.observation_space.low)


space = spaces.Discrete(8)
x = space.sample()

assert space.contains(x)
assert space.n == 8

#env.render()
#
#for i_episode in range(20):
#    observation = env.reset()
#    for t in range(100):
#        env.render()
#        print(observation)
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(action)
#        if done: 
#            print ("Episode finished after {} episodes ".format(t+1))
#            break
            