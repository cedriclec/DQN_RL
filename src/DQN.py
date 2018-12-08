#!/usr/bin/env python3
# title           :DQN.py
# description     :Deep Q-Learning implementation
# author          :Cedric
# date            :07-12-2018
# version         :0.1
# notes           :
# python_version  :3.6.5
# ==========

import gym
import random
import numpy as np
from src.Agent import Agent


class DQN(Agent):
    """Deep Q-Learning Agent"""

    def __init__(self, **kwargs):
        self.algo_name = 'DQN'
        kwargs['log_file'] = self.algo_name + '.log'
        super().__init__(**kwargs)

        self.dnn = self.build_neural_network(self.observation_size, self.action_size)

    def compute_target_reward(self, reward, done, next_obs):
        if not done:
            reward + self.gamma * self.get_max_prediction((self.dnn.predict(next_obs)[0]))
        return reward

    # TODO refactor with DDQN
    # It's the same except the compute target => and the update target in run
    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        # print("in train DQN")
        self.logger.debug("mini_batch ", mini_batch)
        update_input = np.zeros((batch_size, self.observation_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, next_state, action, reward, done = mini_batch[i]
            target = self.dnn.predict(state)[0]

        # TODO Undestand this lines
        ####One DNN
        #for (obs, nextObs, action, reward, done) in mini_batch:

            #targetReward = self.calculTarget(reward, done, nextObs)
            #tar = self.getOutputTraining(targetReward, action, obs)
            #self.dnn.fit(obs, tar, epochs=1, verbose=0)
            #target = self.dnn.predict(state)


            target[action] = self.compute_target_reward(reward, done, next_state)

            update_input[i] = state
            update_target[i] = target

            target = np.reshape(target, [1, target.shape[0]])

            self.dnn.fit(state, target, batch_size=batch_size, epochs=1, verbose=0)

        # TODO understand this line below
        # self.dnn.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

        self.reduce_exploration_randomly()
