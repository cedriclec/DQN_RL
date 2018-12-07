from src import DQN, DDQN

CART_POLE = 'CartPole-v1'
MOUNTAIN_GAME = 'MountainCar-v0'

if __name__ == "__main__":
    kwargs = {'game_name': CART_POLE}
    agent = DQN.DQN(**kwargs)
    agent.run(render=True, nb_episodes=1000)
