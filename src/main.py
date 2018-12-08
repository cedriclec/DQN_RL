from src.DQN import DQN
from src.DDQN import DDQN

CART_POLE = 'CartPole-v1'
MOUNTAIN_GAME = 'MountainCar-v0'

if __name__ == "__main__":
    kwargs = {'game_name': CART_POLE}
    agent = DQN(**kwargs)
    agent.run(render=False, nb_episodes=4000, save_plot=True)
