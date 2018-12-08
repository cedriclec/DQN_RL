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
        self.algo_name = 'DDQN'
        kwargs['log_file'] = self.algo_name + '.log'
        super().__init__(**kwargs)

        self.dnn = self.build_neural_network(self.observation_size, self.action_size)
        self.dnnTarget = self.build_neural_network(self.observation_size, self.action_size)

    def compute_target_reward(self, reward, done, next_obs):
        # Main difference with DQN
        # In DDQN, use target model to give us the reward
        if not done:
            reward = reward + self.discount_factor * self.get_max_prediction((self.dnnTarget.predict(next_obs)[0]))
        return reward

    def update_target_model(self):
        # Update target weights based on those from model
        self.dnnTarget.set_weights(self.dnn.get_weights())
