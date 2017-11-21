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
import gym
import random
import numpy as np
import time
#import tensorflow as tf
from keras import models
from keras.layers import Dense
from collections import deque 
from keras.optimizers import Adam

EPSILON_EXPLORATE = 1.0
MEMORY_MAX_LEN = 2000
NB_HIDDEN_LAYER = 2
NB_NODE_PER_HIDDEN_LAYER = 24
BATCH_SIZE = 48
EPISODE_NB = 1000
TIME_FOR_ONE_EPISODE = 3000

def getMaxPrediction(predictArray):
#    print(predictArray)
#    print(np.amax(predictArray))
    return np.amax(predictArray)

class SmartAgent(object):
    """Agent with deep neural network using tensorflow!"""
    def __init__(self, action_size, observation_size):
        self.epsilonReductor = 0.995
        self.epsilonMinimum = 0.01
        self.learningRate = 0.001
        self.gamma = 0.95
        self.probaToTakeRandomAction = EPSILON_EXPLORATE

        self.action_size = action_size
        self.observation_size = observation_size
        self.dnn = self.buildNeuralNetwork(self.observation_size, self.action_size)
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
            return actionHighestPrediction
        else :
            actionsPrediction = self.dnn.predict(observation)
            actionHighestPrediction = np.argmax(actionsPrediction)
        return actionHighestPrediction

    def actRandomly(self):
        #return random.randrange(self.action_space.n)
        return random.randrange(self.action_size)

    def hasToActRandomly(self):
        return (np.random.rand() <= self.probaToTakeRandomAction)

    def remember(self, observation, nextObservation, action, reward, done):
        self.memory.append((observation, nextObservation, action, reward, done))

    def hasToTrainWithMemory(self):
        return (len(self.memory) > BATCH_SIZE)

    def reduceExplorationRandomly(self):
        if (self.probaToTakeRandomAction > self.epsilonMinimum):
            self.probaToTakeRandomAction *= self.epsilonReductor

    def getOutputTraining(self, target, action, obs):
            tar = self.dnn.predict(obs)
            tar[0][action] = target
            return tar

    def trainWithReplay(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)

        for (obs, nextObs, action, reward, done) in mini_batch:
            targetReward = self.calculTarget(reward, done, nextObs)
            tar = self.getOutputTraining(targetReward, action, obs)
            self.dnn.fit(obs, tar, epochs=1, verbose=0)
            
        self.reduceExplorationRandomly()

    def calculTarget(self, reward, done, nextObs):
        if not done :
            reward = reward + self.gamma * getMaxPrediction(self.dnn.predict(nextObs)[0])
        return reward

    def calcRewardIfNotLastStep(self, reward, done):
        if done:
            reward = -10
        return reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    
    try:
        action_size = env.action_space.n
    except AttributeError:
        action_size = env.action_space.shape[0]
    try:
        observation_size = env.observation_space.n
    except AttributeError:
        observation_size = env.observation_space.shape[0]
        
    agent = SmartAgent(action_size, observation_size)

    for e in range(EPISODE_NB):
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_size])
        for timeSpend in range(TIME_FOR_ONE_EPISODE):
            env.render()
            if ((timeSpend % 1) == 0 ):
                action = agent.act(obs)
    #            print ("Action ", action)
    #            print(type(action))
                nextObs, reward, done, info = env.step(action)
                nextObs = np.reshape(nextObs, [1, observation_size])
            if ((timeSpend % 20) == 0):
                print("Reward : ", reward)
            reward = agent.calcRewardIfNotLastStep(reward, done)
            agent.remember(obs, nextObs, action, reward, done)
            obs = nextObs
#            if done:
#                print("episode: {}/{}, timeForEnd : {}, {}, probToActRandomly: {:.2}"
#                      .format(e, EPISODE_NB, timeSpend, TIME_FOR_ONE_EPISODE, agent.probaToTakeRandomAction))
#                break
        if (agent.hasToTrainWithMemory()):
            agent.trainWithReplay()

    # Close the env and write monitor result info to disk
    env.close()
