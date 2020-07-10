from gym.envs.registration import register

register(
    id="GraphTraffic-v0",
    entry_point='gym_graph_traffic.envs:GraphTrafficEnv'
)
