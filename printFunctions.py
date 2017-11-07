#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:50:26 2017

@author: ced
"""


def printActions(env):
    print("Action available ")
    print(env.action_space)
    
def printObservation(obs):
    print("Observation available ")
    print(obs)
    
def printReward(rew):
    print("Reward get from previous state ")
    print(rew)

