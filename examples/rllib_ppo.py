from gym_graph_traffic.envs.params import PARAMETERS

from datetime import datetime

import ray
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

from gym_graph_traffic.envs.params import PARAMETERS

def env_creator(env_config):
    import gym
    return gym.make('gym_graph_traffic:GraphTraffic-v0', params=env_config)


def run_experiment(params):
    ray.init(num_gpus=1)

    register_env("env", env_creator)

    trainer = ppo.PPOTrainer(env="env", config={
        "env_config": params,

        "lambda": 0.95,
        "lr": 0.0002,
        "train_batch_size": 160,  # default=4000
        "clip_param": 0.2,
        "vf_clip_param": 400000.0,
        "entropy_coeff": 0.001,

        "model": {
            "fcnet_hiddens": [32],
            "fcnet_activation": "relu"
        }
    })

    print("start...")
    start = datetime.now()

    ctr = 0
    for i in range(200):
        results = trainer.train()
        print(f"Train no. {i} completed.")
        print("stats = {}".format(results["hist_stats"]))

    msg = f"end {datetime.now() - start}"
    print(msg)

    ray.shutdown()


if __name__ == '__main__':
    run_experiment(PARAMETERS)
