#!/usr/bin/env python3
# title           :main.py
# description     :Launch agent training
# author          :Cedric
# date            :07-12-2018
# version         :0.1
# notes           :
# python_version  :3.6.5
# ==========

from src.DQN import DQN
from src.DDQN import DDQN

CART_POLE = 'CartPole-v1'
MOUNTAIN_GAME = 'MountainCar-v0'

if __name__ == "__main__":
    kwargs = {'game_name': CART_POLE}
    agent = DQN(**kwargs)
    agent.run(render=False, nb_episodes=5000, save_plot=True)
