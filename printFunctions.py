#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:50:26 2017

@author: ced
"""


def printActions(env):
    printTransition()   
    print (" Action ")
    print ("Action available total ", env.action_space)
    printTransition()
    
def printObservation(obs):
    printTransition()
    print (" OBS ")
    print (obs)
    print ("Shape ", obs.shape)
    print ("Size total ", obs.size)
    printTransition()
    
def printReward(env):
    print("Reward get from previous state ")
    print(env)

def printTransition():
    print (" ======================= ")
    print (" ======================= ")
