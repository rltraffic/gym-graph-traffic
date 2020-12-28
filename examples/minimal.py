import gym

from gym_graph_traffic.envs.params import PARAMETERS

env_name = 'GraphTraffic-v0'
env = gym.make(env_name, params=PARAMETERS)

print("Created environment: {}".format(env_name))

# these parameters can be changed
count_of_steps = 100

for episode in range(20):
    observation = env.reset()

    for step in range(count_of_steps):
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action, count_of_steps)
        print(f"step = {step} action = {action} reward = {reward}")

    env.close()
