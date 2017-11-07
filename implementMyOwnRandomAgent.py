#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:38:56 2017

@author: ced
"""

import argparse
import logging
import sys
import printFunctions
import gym
import numpy as np
import time
import tensorflow as tf
from gym import wrappers
import ten

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        #Choose between random action or action calculate
        self.action_space
        return self.action_space.sample()

    def actRandomly(self):
        return self.action_space.sample()

class SmartAgents(object):
    """Agent with deep neural network using tensorflow!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.dnn = self.buildNeuralNetwork()
        self.probaToTakeRandomAction = 0.2
        self.memory = 2

    def buildNeuralNetwork():
        #use tensor flow for the Neural Network
        return 1
    
    def act(self, observation, reward, done, state):
        if np.random.rand() <= self.probaToTakeRandomAction :
            return self.action_space.sample()
        else :
            actionToChoose = self.dnn.predict(state)
        return self.action_space.sample()
    
    def actRandomly(self):
        return self.action_space.sample()    
        
#        return self.action_space.sample()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
 #   parser.add_argument('env_id', nargs='?', default='SpaceInvaders-v0', help='Select the environment to run')
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    nb_iteration = 1000
    done = False

    printFunctions.printActions(env)
    for i in range(episode_count):
        ob = env.reset()
        done = False
        nb_iteration = 1000
        while (done == False):
        #(nb_iteration > 0):
            env.render()
            action = agent.actRandomly()
            ob, reward, done, info = env.step(action)
            #printFunctions.printReward(reward)
            #printFunctions.printObservation(ob)
            nb_iteration = nb_iteration - 1 
            time.sleep(0.1)
            #time.sleep(1)
            
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()


