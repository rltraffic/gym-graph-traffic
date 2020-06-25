import gym
import gym_graph_traffic

from params import PARAMETERS

env_name = 'GraphTraffic-v0'
env = gym.make(env_name, params=PARAMETERS)

print("Created environment: {}".format(env_name))

for i_episode in range(1):  # 20
    observation = env.reset()

    for t in range(10):  # 100
        action = env.action_space.sample()
        print(action)

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    env.close()
