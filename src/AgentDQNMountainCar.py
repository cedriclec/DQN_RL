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
EPSILON_END = 0.01
MEMORY_MAX_LEN = 5000
BATCH_SIZE = 64
EPISODE_NB = 1000
TIME_FOR_ONE_EPISODE = 3000
MAX_TIME = 300

def getMaxPrediction(predictArray):
    return np.amax(predictArray)

#Todo implement a mother class, and change child class
class DQNAgentMountainCar(object):
    """Agent with deep neural network using tensorflow!"""
    def __init__(self, action_size, observation_size):
        #self.epsilonReductor = 0.995
        self.epsilon = EPSILON_EXPLORATE
        self.epsilonEnd = EPSILON_END
        self.epsilonReductor = (self.epsilon - self.epsilonEnd) / 50000
        self.learningRate = 0.001
        #self.learningRate = 0.00025
        self.gamma = 0.99
        #self.gamma = 0.95

        self.action_size = action_size
        self.observation_size = observation_size
        self.dnn = self.buildNeuralNetwork(self.observation_size, self.action_size)
        self.memory = deque(maxlen=MEMORY_MAX_LEN) #Deque : list-like container with fast appends and pops on either end

    def buildNeuralNetwork(self, inputNodeNumber, outputNodeNumber):
        #use tensor flow Keras Model for building the Neural Network
        model = models.Sequential()
        model.add(Dense(32, input_dim=inputNodeNumber, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(outputNodeNumber, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        return model

    def act(self, observation):
        if  self.hasToActRandomly() :
            actionHighestPrediction = self.actRandomly()
            return actionHighestPrediction
        else :
            actionsPrediction = self.dnn.predict(observation)
            actionHighestPrediction = np.argmax(actionsPrediction)
        return actionHighestPrediction

    def takeRealActionOrFakeAction(self, obs, count_action, previousAction):
        if ((count_action % 4) == 0):
            action = self.act(obs)
            #TODO
            if (action == 0):
                return 0
            elif (action == 1):
                return 2
        return previousAction

    def actRandomly(self):
        #return random.randrange(self.action_space.n)
        return random.randrange(self.action_size)

    def hasToActRandomly(self):
        return (np.random.rand() <= self.epsilon)

    def remember(self, observation, nextObservation, action, reward, done):
        if (action == 2):
            action = 1 #TODO
        self.memory.append((observation, nextObservation, action, reward, done))

    def hasToTrainWithMemory(self):
        return (len(self.memory) > BATCH_SIZE)

    def reduceExplorationRandomly(self):
        if (self.epsilon > self.epsilonEnd):
            self.epsilon -= self.epsilonReductor

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
        if ( (reward != -1) or done):
            print("Reward : ", reward, "  Done : ", done)
#        if done:
#            reward = -10
        return reward

    def getLongerDone(self, timeCount):
        #if (not done):
        done = False
        if timeCount >= MAX_TIME:
            done = True
        return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    game = 'MountainCar-v0'
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    try:
        action_size = env.action_space.n
    except AttributeError:
        action_size = env.action_space.shape[0]
    if(game == 'MountainCar-v0'):
        action_size = 2 #Little hack
    try:
        observation_size = env.observation_space.n
    except AttributeError:
        observation_size = env.observation_space.shape[0]
        
    agent = DQNAgentMountainCar(action_size, observation_size)

    for e in range(EPISODE_NB):
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_size])
        actionCount = 0
        previousAction = 0

        score = 0
        done = False
        while not done:
            obs = env.reset()
            obs = np.reshape(obs, [1, observation_size])
            actionCount += 1
            
            #action = (obs, actionCount)
#            if ((timeSpend % 1) == 0 ):
            action = agent.takeRealActionOrFakeAction(obs, actionCount, previousAction)
            previousAction = action
#            print ("Action ", action)
#            print(type(action))
            nextObs, reward, done, info = env.step(action)
            done = agent.getLongerDone(actionCount)
            nextObs = np.reshape(nextObs, [1, observation_size])
#            if ((timeSpend % 20) == 0):
#                print("Reward : ", reward)
            reward = agent.calcRewardIfNotLastStep(reward, done)
            score += reward

            agent.remember(obs, nextObs, action, reward, done)
            obs = nextObs

            #if (agent.hasToTrainWithMemory()):
            #    agent.trainWithReplay()

            if done:
                print("episode: {}/{}, score : {}, probToActRandomly: {:.2}"
                      .format(e, EPISODE_NB, score, agent.epsilon))
#            if done:
#                print("episode: {}/{}, score : {}, probToActRandomly: {:.2}"
#                      .format(e, EPISODE_NB, score, agent.epsilon))
#        if (agent.hasToTrainWithMemory()):
#            agent.trainWithReplay()

    # Close the env and write monitor result info to disk
    env.close()
