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
from src.config import CART_POLE, MOUNTAIN_GAME

if __name__ == "__main__":
    kwargs = {'game_name': CART_POLE}
    agent = DQN(**kwargs)
    agent.run(render=False, nb_episodes=3000, save_plot=True, nb_episodes_render=50)
