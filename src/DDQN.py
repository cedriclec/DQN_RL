#!/usr/bin/env python3
# title           :DDQN.py
# description     :Double Deep Q-Learning implementation
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


class DDQN(Agent):
    """Double Deep Q-Learning Agent"""

    def __init__(self, **kwargs):
        kwargs['log_file'] = 'DDQN.log'
        super().__init__(**kwargs)

        self.dnn = self.build_neural_network(self.observation_size, self.action_size)
        self.dnnTarget = self.build_neural_network(self.observation_size, self.action_size)

    # TODO refactor with DQN
    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # self.logger.debug("mini_batch %s " % mini_batch)
        update_input = np.zeros((batch_size, self.observation_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, next_state, action, reward, done = mini_batch[i]
            target = self.dnn.predict(state)[0]

            target[action] = self.compute_target(reward, done, next_state)

            update_input[i] = state
            update_target[i] = target

            target = np.reshape(target, [1, target.shape[0]])

            self.dnn.fit(state, target, batch_size=batch_size, epochs=1, verbose=0)
        # TODO understand this line below
        # self.dnn.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

        self.reduce_exploration_randomly()

    def compute_target(self, reward, done, next_obs):
        if not done:
            reward + self.gamma * self.get_max_prediction((self.dnnTarget.predict(next_obs)[0]))
        return reward

    def update_target_model(self):
        # self.dnn.save_weights('weights.h5')
        self.dnnTarget.set_weights(self.dnn.get_weights())
        # self.dnn.load_weights('weights.h5')
