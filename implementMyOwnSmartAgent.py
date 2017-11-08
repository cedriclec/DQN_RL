#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:09:22 2017

@author: ced
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:38:56 2017

@author: ced
"""

import argparse
import logging
import sys
import gym
import random
import numpy as np
#import time
#import tensorflow as tf
#from tensorflow import layers
from keras import models
from keras.layers import Dense
from collections import deque 
from gym import wrappers
from keras.optimizers import Adam

EPSILON_EXPLORATE = 1.0
MEMORY_MAX_LEN = 2000
NB_HIDDEN_LAYER = 2
NB_NODE_PER_HIDDEN_LAYER = 24
BATCH_SIZE = 32
EPISODE_NB = 1000

def getMaxPrediction(predictArray):
#    print(predictArray)
#    print(np.amax(predictArray))
    return np.amax(predictArray)

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

class SmartAgent(object):
    """Agent with deep neural network using tensorflow!"""
    def __init__(self, action_space, observation_space):
        self.epsilonReductor = 0.995
        self.epsilonMinimum = 0.01
        self.learningRate = 0.001
        self.gamma = 0.95
    
        self.action_space = action_space
        self.observation_space = observation_space
        self.dnn = self.buildNeuralNetwork(self.observation_space.shape[0], self.action_space.n)
        self.probaToTakeRandomAction = EPSILON_EXPLORATE
        self.memory = deque(maxlen=MEMORY_MAX_LEN) #Deque : list-like container with fast appends and pops on either end        

    def buildNeuralNetwork(self, inputNodeNumber, outputNodeNumber):
        #use tensor flow Keras Model for building the Neural Network
        model = models.Sequential()
        model.add(Dense(NB_NODE_PER_HIDDEN_LAYER, input_dim=inputNodeNumber, activation='relu')) #24 layer as output
        model.add(Dense(NB_NODE_PER_HIDDEN_LAYER, activation='relu'))
        model.add(Dense(outputNodeNumber, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate)) #TODO
        return model
           
    def act(self, observation):
        if  self.hasToActRandomly() :
            actionHighestPrediction = self.actRandomly()
        else :
            #print("-------------DONT ACT RANDOMLY-----------")
            actionsPrediction = self.dnn.predict(observation)
#            actionHighestPrediction = np.argmax(actionsPrediction)
            actionHighestPrediction = np.argmax(actionsPrediction[0])   
        return actionHighestPrediction

    def remember(self, observation, nextObservation, action, reward, done): #TODO
        self.memory.append((observation, nextObservation, action, reward, done))        

    def hasToActRandomly(self):
        return (np.random.rand() <= self.probaToTakeRandomAction)

    def actRandomly(self):
        return random.randrange(self.action_space.n)
        #return self.action_space.sample()    

    def hasToTrainWithMemory(self):
        return (len(self.memory) > BATCH_SIZE)

    def reduceExplorationRandomly(self):
        if (self.probaToTakeRandomAction > self.epsilonMinimum):
            self.probaToTakeRandomAction *= self.epsilonReductor

    def getOutputTraining(self, target, action, obs):
            tar = self.dnn.predict(obs)
            tar[0][action] = target
            return tar

    def calculTarget(self, reward, done):
        if not done :
            reward = reward + self.gamma * getMaxPrediction(self.dnn.predict(nextObs)[0])
        return reward

    def trainWithReplay(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        for (obs, nextObs, action, reward, done) in mini_batch:
            target = self.calculTarget(reward, done)
            tar = self.getOutputTraining(target, action, obs)
            self.dnn.fit(obs, tar, epochs=1, verbose=0)
            
        self.reduceExplorationRandomly()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
 #   parser.add_argument('env_id', nargs='?', default='SpaceInvaders-v0', help='Select the environment to run')
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    agent = SmartAgent(env.action_space, env.observation_space)
    
    reward = 0
    done = False
    
    for e in range(EPISODE_NB):
        obs = env.reset()
        obs = np.reshape(obs, [1, env.observation_space.shape[0]])
        done = False
        for score in range(500):
            env.render()
            action = agent.act(obs)
            nextObs, reward, done, info = env.step(action)
            nextObs = np.reshape(nextObs, [1, env.observation_space.shape[0]])
            reward = reward if not done else -10
            #Avoid case where finish
            agent.remember(obs, nextObs, action, reward, done)
            obs = nextObs
            if done:
                print("episode: {}/{}, score: {}, probToActRandomly: {:.2}"
                      .format(e, EPISODE_NB, score, agent.probaToTakeRandomAction))
                break
            #time.sleep(1)
        if (agent.hasToTrainWithMemory()):
            agent.trainWithReplay()
        

    # Close the env and write monitor result info to disk
    env.close()