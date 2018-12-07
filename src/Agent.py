#!/usr/bin/env python3
# title           :Agent.py
# description     :This is the abstract class which represents an Agent
# author          :Cedric
# date            :07-12-2018
# version         :0.1
# notes           :
# python_version  :3.6.5
# ==========


from abc import ABC, abstractmethod
import gym
import random
import numpy as np
from collections import deque
import logging
import os

from keras import models
from keras.layers import Dense
from keras.optimizers import Adam

# Game name used
CART_POLE = 'CartPole-v1'
MOUNTAIN_GAME = 'MountainCar-v0'

# Hyper Parameters value
EPSILON_EXPLORATE = 1.0
EPSILON_END = 0.005
MEMORY_MAX_LEN = 10000
BATCH_SIZE = 64
TIME_FOR_ONE_EPISODE = 3000
MAX_TIME = 300
MAX_TIME_FOR_ONE_EPISODE = 300
LEARNING_RATE = 0.001
TRAIN_START_TIME = 1000  # Time to start, it enables to avoid to learn too soon


class Agent(ABC):
    """Agent mother class"""

    def __init__(self, game_name='MountainCar-v0', log_file='agent.log'):
        format = '%(asctime)-15s %(filename)-8s %(levelname)-s %(message)s'
        log_path = 'log'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = os.path.join(log_path, log_file)
        # level = logging.INFO # logging.DEBUG
        level = logging.INFO
        logging.basicConfig(filename=log_path, level=level, format=format)
        self.logger = logging.getLogger()
        self.logger.info("========Init Agent=======")
        self.logger.info("Game used %s" % game_name)
        self.gameName = game_name
        self.env = gym.make(game_name)

        self.trainStart = TRAIN_START_TIME
        self.learningRate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        # self.gamma = 0.99 One in mountainCarDQN
        self.gamma = 0.95
        self.epsilon = EPSILON_EXPLORATE
        self.epsilonEnd = EPSILON_END

        # TODO Check how handle it of general manner
        self.set_init_parameters_wrt_game()

        self.memory = deque(maxlen=MEMORY_MAX_LEN)  # Deque : list-like container with fast appends and pops on either end

    def __del__(self):
        self.env.close()

    def set_init_parameters_wrt_game(self):
        # Todo Implement factory
        if self.gameName == MOUNTAIN_GAME:
            self.action_size = 2
            self.observation_size = self.env.observation_space.shape[0]
        elif self.gameName == CART_POLE:
            self.action_size = self.env.action_space.n
            self.observation_size = self.env.observation_space.shape[0]
        if self.gameName == CART_POLE:
            self.trainStart = self.batch_size
        self.epsilonReductor = (self.epsilon - self.epsilonEnd) / 50000
        if self.gameName == CART_POLE:
            self.epsilonReductor = 0.995

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def compute_target(self, reward, done, next_obs):
        pass

    # Hack : In run only function which differs from DQN / DDQN => FIX IT
    def update_target_model(self):
        pass

    def act_randomly(self):
        return random.randrange(self.action_size)

    def has_to_act_randomly(self):
        return np.random.rand() <= self.epsilon

    # TODO : Change to PyTorch
    def build_neural_network(self, input_node_number, output_node_number):
        # use tensor flow Keras Model for building the Neural Network
        if self.gameName == MOUNTAIN_GAME:
            model = models.Sequential()
            model.add(Dense(32, input_dim=input_node_number, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(output_node_number, activation='linear', kernel_initializer='he_uniform'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        elif self.gameName == CART_POLE:
            model = models.Sequential()
            model.add(Dense(24, input_dim=input_node_number, activation='relu'))  # 24 layer as output
            model.add(Dense(24, activation='relu'))
            model.add(Dense(output_node_number, activation='linear'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))  # TODO
        return model

    def act(self, observation):
        if self.has_to_act_randomly():
            action_highest_prediction = self.act_randomly()
        else :
            actions_prediction = self.dnn.predict(observation)
            action_highest_prediction = np.argmax(actions_prediction[0])
        return action_highest_prediction

    def take_real_action_or_fake_action(self, obs, count_action, previous_action):
        if self.gameName == MOUNTAIN_GAME:
            if (count_action % 4) == 0:
                action = self.act(obs)
                # Action 0 => Left, 2 => Right, 1 => Nothing
                # Don't want our robot to do nothing => TODO put this doc in docs
                if action == 0:
                    previous_action = 0
                elif action == 1:
                    previous_action = 2
        else :
            previous_action = self.act(obs)
        return previous_action

    def remember(self, observation, next_observation, action, reward, done):
        if self.gameName == MOUNTAIN_GAME:
            if action == 2:
                action = 1 # TODO : Voir le pourquoi du mountain-game, l'action 2 doit être changée en 1
        self.memory.append((observation, next_observation, action, reward, done))

    def has_to_train_with_memory(self):
        # self.logger.debug("memory : %s, batch_size : %s, train_start : %s" %(len(self.memory), self.batch_size, self.trainStart))
        return (len(self.memory) > self.batch_size) and (len(self.memory) > self.trainStart)

    def reduce_exploration_randomly(self):
        if self.gameName == MOUNTAIN_GAME:
            if self.epsilon > self.epsilonEnd:
                self.epsilon -= self.epsilonReductor
        else:
            self.epsilon *= self.epsilonReductor

    def get_output_training(self, target, action, obs):
            tar = self.dnn.predict(obs)
            tar[0][action] = target
            return tar

    def calc_reward_if_not_last_step(self, reward, done, time):
        # Fail to converge to goal
        # Try to give bad bad reward if don't reach the goal
        # And good reward if reach the goal

        # TODO : Factory
        if (self.gameName == MOUNTAIN_GAME) and done and (time < MAX_TIME_FOR_ONE_EPISODE ):
            reward = 100
        elif (self.gameName == CART_POLE) and done:
            reward = -10
        return reward

    def get_longer_done(self, time_count, obs, done):
        #  if not done:
        # self.logger.debug("obs %d" % obs[0])
        if self.gameName == MOUNTAIN_GAME:
            done = False
            if (time_count >= MAX_TIME) or (obs[0] > 0.5):
                done = True
        return done

    @staticmethod
    def get_max_prediction(predict_array):
        return np.amax(predict_array)

    def update_to_save_model(self):
        self.dnn.load_weights('weights.h5')

    def run(self, nb_episodes=20, render=False):
        self.logger.info('*'*10)
        self.logger.info("Start training for %s episodes" % nb_episodes)
        self.logger.info('*'*10)

        scores, episodes = [], []

        for e in range(nb_episodes):
            obs = self.env.reset()
            obs = np.reshape(obs, [1, self.observation_size])

            actionCount = 0
            previous_action = 0

            score = 0
            done = False

            while not done:
                if render:
                    self.env.render()

                actionCount += 1

                action = self.take_real_action_or_fake_action(obs, actionCount, previous_action)
                previous_action = action
                # self.logger.debug("action %d " % action)
                next_obs, reward, done, info = self.env.step(action)
                done = self.get_longer_done(actionCount, next_obs, done)
                reward = self.calc_reward_if_not_last_step(reward, done, actionCount)
                next_obs = np.reshape(next_obs, [1, self.observation_size])
                # self.logger.debug("next_obs %s " % next_obs)

                score += reward
                self.remember(obs, next_obs, action, reward, done)

                # Train at every action => TODO Check if should really be done
                if (self.gameName == MOUNTAIN_GAME) and self.has_to_train_with_memory():
                    self.train()

                obs = next_obs

                if done:
                    self.env.reset()
                    self.update_target_model() # Find a way to avoid this one

                    scores.append(score)
                    episodes.append(e)

                    self.logger.info("episode: %s, score %s, memory length : %s, epsilon : %s"
                                     %(e, score, len(self.memory), self.epsilon))

            # TODO check if this one is better => train only at the end
            if (self.gameName == CART_POLE) and self.has_to_train_with_memory():
                self.train()

        self.logger.info('/' * 10)
        self.logger.info("End of the training")
        self.logger.info('/' * 10)
