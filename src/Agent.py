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
from tqdm import trange
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense
from keras.optimizers import Adam
from src.config import CART_POLE, MOUNTAIN_GAME

# Hyper Parameters value
EPSILON_EXPLORATE = 1.0
EPSILON_END = 0.005
MEMORY_MAX_LEN = 10000
BATCH_SIZE = 64
TIME_FOR_ONE_EPISODE = 3000
MAX_TIME = 300
MAX_TIME_FOR_ONE_EPISODE = 300
LEARNING_RATE = 0.01  # 0.001
TRAIN_START_TIME = 1000  # Time to start, it enables to avoid to learn too soon


class Agent(ABC):
    """Agent mother class"""

    # region Init
    def __init__(self, game_name='MountainCar-v0', log_file='agent.log'):
        format = '%(asctime)-15s %(levelname)-s %(message)s'
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
        self.discount_factor = 0.99 # 0.95
        self.epsilon = EPSILON_EXPLORATE
        self.epsilon_end = EPSILON_END
        # self.epsilon_reductor = (self.epsilon - self.epsilon_end) / 50000
        # self.observation_size = self.env.observation_space.shape[0]
        self.set_init_parameters_wrt_game()

        self.memory = deque(maxlen=MEMORY_MAX_LEN)  # Deque : list-like container with fast appends and pops on either end

    def __del__(self):
        self.env.close()

    def set_init_parameters_wrt_game(self):
        if self.gameName == MOUNTAIN_GAME:
            self.action_size = 2
            self.observation_size = self.env.observation_space.shape[0]
        elif self.gameName == CART_POLE:
            self.action_size = self.env.action_space.n
            self.observation_size = self.env.observation_space.shape[0]
        if self.gameName == CART_POLE:
            self.trainStart = self.batch_size
        self.epsilon_reductor = (self.epsilon - self.epsilon_end) / 50000
        if self.gameName == CART_POLE:
            self.epsilon_reductor = 0.995

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
            model.add(Dense(24, input_dim=input_node_number, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(output_node_number, activation='linear', kernel_initializer='he_uniform'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        return model
    # endregion

    # region ACT
    # ============================ACT===================

    # Hack : In run only function which differs from DQN / DDQN
    def update_target_model(self):
        pass

    def act_randomly(self):
        return random.randrange(self.action_size)

    def has_to_act_randomly(self):
        return np.random.rand() <= self.epsilon

    def act(self, observation):
        if self.has_to_act_randomly():
            action_highest_prediction = self.act_randomly()
        else:
            actions_prediction = self.dnn.predict(observation)
            action_highest_prediction = np.argmax(actions_prediction[0])
        return action_highest_prediction

    def take_real_action_or_fake_action(self, obs, count_action, previous_action):
        if self.gameName == MOUNTAIN_GAME:
            if (count_action % 4) == 0:
                action = self.act(obs)
                # Action 0 => Left, 2 => Right, 1 => Nothing
                # Don't want our robot to do nothing =>
                if action == 0:
                    previous_action = 0
                elif action == 1:
                    previous_action = 2
        else:
            previous_action = self.act(obs)
        return previous_action
    # endregion

    # region train
    # ==========================TRAIN================
    # It's the same except the compute target => and the update target in run
    def train(self):
        # Get some random episode from memory, and train on this episodes
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        self.logger.debug("mini_batch ", mini_batch)
        update_input = np.zeros((batch_size, self.observation_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, next_state, action, reward, done = mini_batch[i]
            target = self.dnn.predict(state)[0]

            # Associate to the action used in memory, the reward associated
            target[action] = self.compute_target_reward(reward, done, next_state)
            update_input[i] = state
            update_target[i] = target

        # Train on all batch update
        self.dnn.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

        # After each train, reduce the exploration randomly
        self.reduce_exploration_randomly()

    @abstractmethod
    def compute_target_reward(self, reward, done, next_obs):
        pass

    def remember(self, observation, next_observation, action, reward, done):
        if self.gameName == MOUNTAIN_GAME:
            if action == 2:
                action = 1
        self.memory.append((observation, next_observation, action, reward, done))

    def has_to_train_with_memory(self):
        # self.logger.debug("memory : %s, batch_size : %s, train_start : %s" %(len(self.memory), self.batch_size, self.trainStart))
        return (len(self.memory) > self.batch_size) and (len(self.memory) > self.trainStart)

    def reduce_exploration_randomly(self):
        if self.gameName == MOUNTAIN_GAME:
            if self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_reductor
        else:
            self.epsilon *= self.epsilon_reductor

    def calc_reward_if_not_last_step(self, reward, done, time):
        # Fail to converge to goal
        # Try to give bad bad reward if don't reach the goal
        # And good reward if reach the goal

        if (self.gameName == MOUNTAIN_GAME) and done and (time < MAX_TIME_FOR_ONE_EPISODE ):
            reward = 100
        elif (self.gameName == CART_POLE) and done:
            reward = -10
        return reward

    def run(self, nb_episodes=20, render=False, save_plot=False, nb_episodes_render=100):
        self.logger.info('*'*10)
        self.logger.info("Start training for %s episodes" % nb_episodes)
        self.logger.info('*'*10)

        scores, episodes, epsilons = [], [], []

        for e in trange(nb_episodes):
            obs = self.env.reset()
            obs = np.reshape(obs, [1, self.observation_size])

            actionCount = 0
            previous_action = 0

            score = 0
            done = False

            while not done:
                if nb_episodes_render >= nb_episodes or nb_episodes_render <= 0:
                    nb_episodes_render = 1
                if render or (e % (nb_episodes / nb_episodes_render) == 0):
                    self.env.render()

                actionCount += 1

                action = self.take_real_action_or_fake_action(obs, actionCount, previous_action)
                previous_action = action
                # self.logger.debug("action %d " % action)
                next_obs, reward, done, _ = self.env.step(action)
                done = self.get_longer_done(actionCount, next_obs, done)
                reward = self.calc_reward_if_not_last_step(reward, done, actionCount)
                next_obs = np.reshape(next_obs, [1, self.observation_size])
                # self.logger.debug("next_obs %s " % next_obs)

                score += reward
                self.remember(obs, next_obs, action, reward, done)

                # Train at every action => Mountain game need a lot of train (caus lot of examples never reach the goal)
                if (self.gameName == MOUNTAIN_GAME) and self.has_to_train_with_memory():
                    self.train()

                obs = next_obs

                if done:
                    self.env.reset()
                    self.update_target_model()  # Find a way to avoid this one

                    scores.append(score)
                    episodes.append(e)
                    epsilons.append(self.epsilon)

                    self.logger.info("episode: %s, score %s, memory length : %s, epsilon : %s"
                                     %(e, score, len(self.memory), self.epsilon))

            if (self.gameName == CART_POLE) and self.has_to_train_with_memory():
                self.train()

        if save_plot:
            kwargs = {
                'scores': scores,
                'episodes': episodes,
                'epsilons': epsilons
            }
            self.plot_stat(**kwargs, save_plot=True)

        self.logger.info('/' * 10)
        self.logger.info("End of the training")
        self.logger.info('/' * 10)
    # endregion

    # region plot stats
    def mean_some_range(self, x, y, nb_split=100):
        x_split = np.array_split(x, nb_split)
        y_split = np.array_split(y, nb_split)
        x_mean = np.fromiter(map(np.mean, x_split), dtype=np.int16)
        y_mean = np.fromiter(map(np.mean, y_split), dtype=np.float32)
        return x_mean, y_mean

    def plot_stat(self, scores, episodes, epsilons, save_plot=False):
        date = str(datetime.now()) # Using datetime.now() directly add a point in string
        nb_split = 200 # 100

        episodes_mean, scores_mean = self.mean_some_range(episodes, scores, nb_split)
        plt.plot(episodes_mean, scores_mean)
        plt.title("Average Score evolution for %s " % self.algo_name)
        plt.ylabel("Average Score at the end of episode")
        plt.xlabel("episode number")
        if save_plot:
            file_save = self.algo_name + '_score_' + date + '.png'
            file_save = os.path.join('log', file_save)
            plt.savefig(file_save, format='png')
        plt.show(block=True)

        episodes_mean, epsilons_mean = self.mean_some_range(episodes, epsilons, nb_split)
        plt.plot(episodes_mean, epsilons_mean)
        plt.title("AverageEpsilon evolution for %s " % self.algo_name)
        plt.ylabel("Average Epsilon during the episode")
        plt.xlabel("Episode number")
        if save_plot:
            file_save = self.algo_name + '_epsilon_' + date + '.png'
            file_save = os.path.join('log', file_save)
            plt.savefig(file_save, format='png')
        plt.show(block=True)
    # endregion

    # region tool
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
    # endregion
