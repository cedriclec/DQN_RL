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
from keras import models
from keras.layers import Dense
from collections import deque 
from keras.optimizers import Adam

MOUNTAIN_GAME = 'MountainCar-v0'
EPSILON_EXPLORATE = 1.0
EPSILON_END = 0.005
MEMORY_MAX_LEN = 10000
BATCH_SIZE = 64
TIME_FOR_ONE_EPISODE = 3000
MAX_TIME = 300
MAX_TIME_FOR_ONE_EPISODE = 300
LEARNING_RATE = 0.001
TRAIN_START_TIME = 1000 #Time to start avoid learn too soon,
RENDER = False

CART_POLE = 'CartPole-v1'

def getMaxPrediction(predictArray):
    return np.amax(predictArray)

#Todo implement a mother class, and change child class
class DDQNAgentMountainCar(object):
    """Agent with deep neural network using tensorflow!"""
    #def __init__(self, action_size, observation_size):
    #TODO Improve this one
    def __init__(self, action_size = -1, observation_size = -1, gameName = 'MountainCar-v0'):

        self.action_size = action_size
        self.observation_size = observation_size
        self.gameName = gameName
        self.env = gym.make(gameName)

        self.setInitParams()

        self.render = RENDER
        self.trainStart = TRAIN_START_TIME
        if (self.gameName == CART_POLE):
            self.trainStart = BATCH_SIZE
        self.learningRate = LEARNING_RATE
        #self.learningRate = 0.00025

        #self.gamma = 0.99 One in mountainCarDQN
        self.gamma = 0.95

        self.epsilon = EPSILON_EXPLORATE
        self.epsilonEnd = EPSILON_END
        self.epsilonReductor = (self.epsilon - self.epsilonEnd) / 50000
        if (self.gameName == CART_POLE):
            self.epsilonReductor = 0.995
        self.dnn = self.buildNeuralNetwork(self.observation_size, self.action_size)
        self.dnnTarget = self.buildNeuralNetwork(self.observation_size, self.action_size)
        self.memory = deque(maxlen=MEMORY_MAX_LEN) #Deque : list-like container with fast appends and pops on either end

    def __del__(self):
        self.env.close()

    def setInitParams(self):
        #Todo Implement factory
        if( (self.action_size == -1) or (self.observation_size == -1) ):
            if (self.gameName == MOUNTAIN_GAME):
                self.action_size = 2
                self.observation_size = self.env.observation_space.shape[0]
            elif (self.gameName == CART_POLE):
                self.action_size = self.env.action_space.n
                self.observation_size = self.env.observation_space.shape[0]


    def buildNeuralNetwork(self, inputNodeNumber, outputNodeNumber):
        #use tensor flow Keras Model for building the Neural Network
        if (self.gameName == MOUNTAIN_GAME):
            model = models.Sequential()
            model.add(Dense(32, input_dim=inputNodeNumber, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(outputNodeNumber, activation='linear', kernel_initializer='he_uniform'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        elif (self.gameName == CART_POLE):
            model = models.Sequential()
            model.add(Dense(24, input_dim=inputNodeNumber, activation='relu'))  # 24 layer as output
            model.add(Dense(24, activation='relu'))
            model.add(Dense(outputNodeNumber, activation='linear'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))  # TODO
        return model

    def act(self, observation):
        if self.hasToActRandomly():
            actionHighestPrediction = self.actRandomly()
        else :
            actionsPrediction = self.dnn.predict(observation)
            actionHighestPrediction = np.argmax(actionsPrediction[0])
        return actionHighestPrediction

    def takeRealActionOrFakeAction(self, obs, count_action, previousAction):
        if (self.gameName == MOUNTAIN_GAME):
            if ((count_action % 4) == 0):
                action = self.act(obs)
                #Action 0 => Left, 2 => Right, 1 => Nothing
                #Don't want our robot to do nothing
                if (action == 0):
                    previousAction = 0
                elif (action == 1):
                    previousAction = 2
        else :
            previousAction = self.act(obs)
        return previousAction

    def actRandomly(self):
        return random.randrange(self.action_size)

    def hasToActRandomly(self):
        return (np.random.rand() <= self.epsilon)

    def remember(self, observation, nextObservation, action, reward, done):
        if (self.gameName == MOUNTAIN_GAME):
            if (action == 2):
                action = 1 #TODO
        self.memory.append((observation, nextObservation, action, reward, done))

    def hasToTrainWithMemory(self):
        #print("memory ", len(self.memory), " BATCH_SIZE ", BATCH_SIZE ," self.trainStart ", self.trainStart)
        return ( (len(self.memory) > BATCH_SIZE) and (len(self.memory) > self.trainStart) )

    def reduceExplorationRandomly(self):
        if (self.gameName == MOUNTAIN_GAME):
            if (self.epsilon > self.epsilonEnd):
                self.epsilon -= self.epsilonReductor
        else :
            self.epsilon *= self.epsilonReductor

    def trainWithReplay(self):
        #TODO Refactor this one
        batchSize = min(BATCH_SIZE, len(self.memory))
        mini_batch = random.sample(self.memory, batchSize)

        #print("mini_batch ", mini_batch)
        updateInput = np.zeros((batchSize, self.observation_size))
        updateTarget = np.zeros((batchSize, self.action_size))

        for i in range(batchSize):
            state, next_state, action, reward, done = mini_batch[i]
            target = self.dnn.predict(state)[0]

        ####One DNN
        #for (obs, nextObs, action, reward, done) in mini_batch:

            #targetReward = self.calculTarget(reward, done, nextObs)
            #tar = self.getOutputTraining(targetReward, action, obs)
            #self.dnn.fit(obs, tar, epochs=1, verbose=0)
            #target = self.dnn.predict(state)


            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * getMaxPrediction(self.dnnTarget.predict(next_state)[0])

                #Use one DNN
                #target[action] = reward + self.gamma * getMaxPrediction(self.dnn.predict(next_state)[0])

            target = np.reshape(target, [1, target.shape[0]])
            updateInput[i] = state
            updateTarget[i] = target

            #self.dnn.fit(state, target, batch_size=batchSize, epochs=1, verbose=0)
        self.dnn.fit(updateInput, updateTarget, batch_size=batchSize, epochs=1, verbose=0)

        self.reduceExplorationRandomly()

    def calculTarget(self, reward, done, nextObs):
        if not done :
            reward = reward + self.gamma * getMaxPrediction(self.dnn.predict(nextObs)[0])
        return reward

    def getOutputTraining(self, target, action, obs):
            tar = self.dnn.predict(obs)
            tar[0][action] = target
            return tar

    def calcRewardIfNotLastStep(self, reward, done, time):
        #Fail to converge to goal
        #Try to give bad bad reward if don't reach the goal
        #And good reward if reach the goal

        if ( (self.gameName == MOUNTAIN_GAME) and done and (time < MAX_TIME_FOR_ONE_EPISODE ) ):
            reward = 10 #100 Before
        elif ((self.gameName == CART_POLE) and done):
            reward = -10
        return reward

    def getLongerDone(self, timeCount, obs, done):
        if (self.gameName == MOUNTAIN_GAME):
            done = False
            if (timeCount >= MAX_TIME) or (obs[0] > 0.5):
                done = True
        return done

    def updateTargetModel(self):
        #self.dnn.save_weights('weights.h5')
        self.dnnTarget.set_weights(self.dnn.get_weights())
        #self.dnn.load_weights('weights.h5')

    def updateToSaveModel(self):
        self.dnn.load_weights('weights.h5')

    def run(self, render = False, nbEpisodes= 20):

        scores, episodes = [], []

        for e in range(nbEpisodes):
            obs = self.env.reset()
            obs = np.reshape(obs, [1, self.observation_size])

            actionCount = 0
            previousAction = 0

            score = 0
            done = False

            while not done:
                if ( (self.render) or (render) ):
                    self.env.render()

                actionCount += 1

                action = self.takeRealActionOrFakeAction(obs, actionCount, previousAction)
                previousAction = action
                nextObs, reward, done, info = self.env.step(action)
                done = self.getLongerDone(actionCount, nextObs, done)
                reward = self.calcRewardIfNotLastStep(reward, done, actionCount)
                nextObs = np.reshape(nextObs, [1, self.observation_size])

                score += reward
                self.remember(obs, nextObs, action, reward, done)

                if ((self.gameName == MOUNTAIN_GAME) and self.hasToTrainWithMemory()):
                    self.trainWithReplay()

                obs = nextObs

                if done:
                    self.env.reset()
                    self.updateTargetModel()

                    scores.append(score)
                    episodes.append(e)

                    print("episode:", e, "  score:", score, "  memory length:", len(self.memory),
                          "  epsilon:", self.epsilon)

            if ((self.gameName == CART_POLE) and self.hasToTrainWithMemory()):
                self.trainWithReplay()
