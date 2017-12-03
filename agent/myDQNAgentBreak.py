#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:47:46 2017

@author: ced
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:09:22 2017

@author: ced
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:38:56 2017

@author: ced
"""

import argparse
import gym
import random
import numpy as np
from toolFunction import convert3DColorTo2DGrey

import printFunctions
from keras.models import Model, Input
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import losses

# Constant

INITIAL_EPSILON_EXPLORATE = 1.0
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1e6  # TODO

IMG_WIDTH = 48  # TODO Not take this one
IMG_HEIGHT = 48  # TODO Not take this one
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_OF_ACTIONS = 4

AGENT_HISTORY_LENGTH = 4
ACTION_REPEAT = 4  # TODO
UPDATE_FREQUENCY = 4  # TODO
LEARNING_RATE = 0.000025

MEMORY_MAX_LEN = 1e6  # 400000
REPLAY_START_SIZE = 50000

BATCH_SIZE = 32
EPISODE_NB = 50000
TIME_FOR_ONE_EPISODE = 3000  # TODO Delete maybe
TARGET_NETWORK_UPDATE_FREQUENCY = 1e4  # TODO Learn this one
DISCOUNT_FACTOR = 0.99


def getMaxPrediction(predictArray):
    return np.amax(predictArray)


class SmartAgent(object):
    """Agent with deep neural network using tensorflow!"""

    def __init__(self, action_size, observation_size):

        self.action_size = action_size
        self.observation_size = (84, 84, 4)

        self.epsilonStart = INITIAL_EPSILON_EXPLORATE
        self.epsilon = self.epsilonStart
        self.epsilonEnd = FINAL_EXPLORATION
        self.epsilonReductor = (self.epsilonStart - self.epsilonEnd) / 1000000

        self.batch_size = BATCH_SIZE

        self.learningRate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR  # Discount factor

        self.nbMaxActionAtInit = 30

        self.model = self.buildNeuralNetwork(self.observation_size, self.action_size)
        self.modelCopy = self.copy_model(self.model)
        self.memory = deque(maxlen=MEMORY_MAX_LEN)  # Deque : list-like container with fast appends and pops on either end

        self.optimizer = self.optimizer()

    def copy_model(self, model):
        model.save_weights('weights.h5')
        new_model = self.get_model()
        new_model.load_weights('weights.h5')
        return new_model

    def buildNeuralNetwork(self, inputNodeNumber, outputNodeNumber):
        # use tensor flow Keras Model for building the Neural Network

        # TODO May be use the BatchNormalization
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=inputNodeNumber))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(outputNodeNumber))
        model.compile(Adam(self.LEARNING_RATE), loss=losses.mean_squared_error, metrics=['accuracy'])
        model.summary()
        return model

    def act(self, observation):
        if self.hasToActRandomly():
            actionHighestPrediction = self.actRandomly()
            actionHighestPrediction
        else:
            qValue = self.model.predict(observation)
            actionHighestPrediction = np.argmax(qValue)
        return actionHighestPrediction


    def actRandomly(self):
        # return random.randrange(self.action_space.n)
        return random.randrange(self.action_size)

    def hasToActRandomly(self):
        return (np.random.rand() <= self.epsilon)

    def remember(self, history, nextHistory, action, reward, done):
        self.memory.append((history, nextHistory, action, reward, done))

    def hasToTrainWithMemory(self):
        return (len(self.memory) > (MEMORY_MAX_LEN / 10))

    def reduceExplorationRandomly(self):
        if (self.epsilon > self.epsilonMinimum):
            self.epsilon *= self.epsilonReductor

    def getOutputTraining(self, target, action, obs):
        tar = self.model.predict(obs)
        tar[0][action] = target
        return tar

    def trainWithReplay(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        targetValue = self.calculTarget(reward, dead, next_history)
        for (obs, nextObs, action, reward, done) in mini_batch:
            targetReward = self.calculTarget(reward, done, nextObs)
            tar = self.getOutputTraining(targetReward, action, obs)
            self.model.fit(obs, tar, epochs=1, verbose=0)

        self.reduceExplorationRandomly()

    def calculTarget(self, reward, done, nextObs):
        for i in range(self.batch_size)
            if not done:
                reward = reward + self.gamma * getMaxPrediction(self.model.predict(nextObs)[0])
        return reward

    def calcRewardIfNotLastStep(self, reward, done):
        if done:
            reward = -10
        return reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    parser.add_argument('env_id', nargs='?', default='Breakout-v4', help='Select the environment to run')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    try:
        action_size = env.action_space.n
    except AttributeError:
        action_size = env.action_space.shape[0]
    try:
        observation_size = env.observation_space.n
    except AttributeError:
        observation_size = env.observation_space.shape[0] * env.observation_space.shape[
            1]  # TODO Calculate size in a function
        # observation_size = env.observation_space.shape[0]
    #       observation_size = env.observation_space
    #       print(observation_size)

    agent = SmartAgent(action_size, observation_size)
    # printFunctions.printActions(env)
    for e in range(EPISODE_NB):
        #TODO Add if want to print
        obs = env.reset()
        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        #for _ in range(random.randint(1, agent.nbMaxActionAtInit)):
        #    observe, _, _, _ = env.step(1)  # DO nothing

        # printFunctions.printObservation(obs)
        obs = convert3DColorTo2DGrey(obs)
        history = [obs]*AGENT_HISTORY_LENGTH
        done = False
        # printFunctions.printObservation(obs1D)
        done = 0
        action = 0
        steps = 0
        while not done :
            env.render()
            if ((steps % ACTION_REPEAT) == 0):
                action = agent.act(obs)

                nextObs, reward, done, info = env.step(action)
                nextObs = convert3DColorTo2DGrey(nextObs)
                nextObs = np.reshape([nextObs], (1, 84, 84, 1))  # TODO Work on this one

                nextHistory = np.append(nextObs, obs[:, :, :, :3], axis=3)

                reward = agent.calcRewardIfNotLastStep(reward, done)
            agent.remember(obs, nextObs, action, reward, done)
            obs = nextObs
            if done:
                print("episode: {}/{}, timeForEnd : {}, {}, probToActRandomly: {:.2}"
                      .format(e, EPISODE_NB, timeSpend, TIME_FOR_ONE_EPISODE, agent.epsilon))
                break
        if (agent.hasToTrainWithMemory()):
            agent.trainWithReplay()

    # Close the env and write monitor result info to disk
    env.close()
