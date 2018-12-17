#!/usr/bin/env python3
# title           :DQN.py
# description     :Deep Q-Learning implementation
# author          :Cedric
# date            :07-12-2018
# version         :0.1
# notes           :
# python_version  :3.6.5
# ==========

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
            reward = reward + self.discount_factor * self.get_max_prediction((self.dnn.predict(next_obs)[0]))
        return reward
