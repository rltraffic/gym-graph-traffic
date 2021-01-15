import gym

from gym_graph_traffic.envs.params import PARAMETERS
from gym_graph_traffic.envs.graph_traffic import GraphTrafficEnv


env_name = 'GraphTraffic-v0'
env = gym.make(env_name, params=PARAMETERS)

print("Created environment: {}".format(env_name))

for episode in range(1):  # 20
    observation = env.reset()

    for step in range(10):  # 100
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        print(f"step = {step} action = {action} reward = {reward}")

    env.close()


# save data in file
if PARAMETERS.save_count_of_cars and "grid" in PARAMETERS.preset_name:
    GraphTrafficEnv._save_data(PARAMETERS, env.intersections)
