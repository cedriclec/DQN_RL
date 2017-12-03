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

EPSILON_EXPLORATE = 1.0
EPSILON_END = 0.01
MEMORY_MAX_LEN = 5000
BATCH_SIZE = 64
EPISODE_NB = 1000
TIME_FOR_ONE_EPISODE = 3000
MAX_TIME = 300
LEARNING_RATE = 0.001
TRAIN_START_TIME = 1000 #Time to start avoid learn too soon,
RENDER = False


def getMaxPrediction(predictArray):
    return np.amax(predictArray)

#Todo implement a mother class, and change child class
class DDQNAgentMountainCar(object):
    """Agent with deep neural network using tensorflow!"""
    #def __init__(self, action_size, observation_size):
    #TODO Improve this one
    def __init__(self, action_size = -1, observation_size = -1, gameName = 'MountainCar-v0'):
        #self.epsilonReductor = 0.995

        self.action_size = action_size
        self.observation_size = observation_size

        self.setInitParams(gameName)

        self.render = RENDER
        self.trainStart = TRAIN_START_TIME

        self.learningRate = LEARNING_RATE
        #self.learningRate = 0.00025
        self.gamma = 0.99
        #self.gamma = 0.95

        self.epsilon = EPSILON_EXPLORATE
        self.epsilonEnd = EPSILON_END
        self.epsilonReductor = (self.epsilon - self.epsilonEnd) / 50000


        self.dnn = self.buildNeuralNetwork(self.observation_size, self.action_size)
        self.dnnTarget = self.buildNeuralNetwork(self.observation_size, self.action_size)
        self.memory = deque(maxlen=MEMORY_MAX_LEN) #Deque : list-like container with fast appends and pops on either end

    def setInitParams(self, gameName):
        if( (self.action_size == -1) or (self.observation_size == -1) ):
            if (gameName == 'MountainCar-v0'):
                env = gym.make(gameName)
                self.action_size = 2
                self.observation_size = env.observation_space.shape[0]

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
        return random.randrange(self.action_size)

    def hasToActRandomly(self):
        return (np.random.rand() <= self.epsilon)

    def remember(self, observation, nextObservation, action, reward, done):
        if (action == 2):
            action = 1 #TODO
        self.memory.append((observation, nextObservation, action, reward, done))

    def hasToTrainWithMemory(self):
        #print("memory ", len(self.memory), " BATCH_SIZE ", BATCH_SIZE ," self.trainStart ", self.trainStart)
        return ( (len(self.memory) > BATCH_SIZE) and (len(self.memory) > self.trainStart) )

    def reduceExplorationRandomly(self):
        if (self.epsilon > self.epsilonEnd):
            self.epsilon -= self.epsilonReductor

    def trainWithReplay(self):
        #TODO Refactor this one
        batchSize = min(BATCH_SIZE, len(self.memory))
        mini_batch = random.sample(self.memory, batchSize)

        updateInput = np.zeros((batchSize, self.observation_size))
        updateTarget = np.zeros((batchSize, self.action_size))

        for i in range(batchSize):
            state, next_state, action, reward, done = mini_batch[i]
            target = self.dnn.predict(state)[0]

            # 큐러닝에서와 같이 s'에서의 최대 Q Value를 가져옴. 단, 타겟 모델에서 가져옴
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                                          np.argmax(self.dnnTarget.predict(next_state))
            updateInput[i] = state
            updateTarget[i] = target

        self.dnn.fit(updateInput, updateTarget, batch_size=batchSize, epochs=1, verbose=0)

            
        self.reduceExplorationRandomly()

    def calculTarget(self, reward, done, nextObs):
        if not done :
            reward = reward + self.gamma * getMaxPrediction(self.dnn.predict(nextObs)[0])
        return reward

    def getOutputTraining(self, target, action, obs):
            tar = self.dnn.predict(obs)
            tar[action] = target
            return tar

    def calcRewardIfNotLastStep(self, reward, done):
#        if done:
#            reward = -10
        return reward

    def getLongerDone(self, timeCount):
        #if (not done):
        done = False
        if timeCount >= MAX_TIME:
            done = True
        return done

    def updateTargetModel(self):
        self.dnn.save_weights('weights.h5')
        self.dnnTarget.set_weights(self.dnn.get_weights())

    def run(self,  render = False, nbEpisodes= 20, gameName = 'MountainCar-v0'):
        env = gym.make(gameName)
        #try:
        #    action_size = env.action_space.n
        #except AttributeError:
         #   action_size = env.action_space.shape[0]
        #if(gameName == 'MountainCar-v0'):
        action_size = 2 #Little hack
        #try:
         #   observation_size = env.observation_space.n
        #except AttributeError:
         #   observation_size = env.observation_space.shape[0]

        #observation_size = env.observation_space.shape[0]
        #agent = DDQNAgentMountainCar(action_size, observation_size)

        for e in range(nbEpisodes):
            obs = env.reset()
            obs = np.reshape(obs, [1, self.observation_size])

            actionCount = 0
            previousAction = 0

            score = 0
            done = False

            while not done:
                if (self.render or render):
                    env.render()

                actionCount += 1

                #action = (obs, actionCount)
    #            if ((timeSpend % 1) == 0 ):
                action = self.takeRealActionOrFakeAction(obs, actionCount, previousAction)
                previousAction = action
    #            print ("Action ", action)
    #            print(type(action))
                nextObs, reward, done, info = env.step(action)
                #done = agent.getLongerDone(actionCount)
                nextObs = np.reshape(nextObs, [1, self.observation_size])

                score += reward
                reward = self.calcRewardIfNotLastStep(reward, done)
                self.remember(obs, nextObs, action, reward, done)
                obs = nextObs
                if (self.hasToTrainWithMemory()):
                    #print("Start Train", agent.hasToTrainWithMemory())
                    self.trainWithReplay()

                if done:
                    print("episode: {}/{}, score : {}, probToActRandomly: {:.2}"
                          .format(e, nbEpisodes, score, self.epsilon))
                    self.updateTargetModel()


      #      if (agent.hasToTrainWithMemory()):
     #            agent.trainWithReplay()

        # Close the env and write monitor result info to disk
        env.close()

