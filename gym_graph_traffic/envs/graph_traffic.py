from typing import List

import gym
import numpy as np
import pygame
from attrdict import AttrDict

from gym_graph_traffic.envs.intersection import FourWayNoTurnsIntersection
from gym_graph_traffic.envs.segment import Segment

RETURN_CHARIOT = "\u001b[0;0H"


class GraphTrafficEnv(gym.Env):
    def __init__(self, params):

        print(params)

        params = AttrDict(params)
        self.params = params

        # simulation params
        self.updates_per_step = params.traffic_graph.updates_per_step
        self.max_steps = params.traffic_graph.max_steps
        self.offsets = params.traffic_graph.offsets
        self.offsets_raw = params.traffic_graph.offsets_raw
        self.num_offsets = len(params.traffic_graph.offsets)

        # road graph
        self.num_intersections = len(params.traffic_graph.intersections)
        self.num_segments = len(params.traffic_graph.segments)
        self.intersections = []
        self.segments = []
        self._set_up_road_graph(params)

        # current simulation status
        self.current_step = 0
        self.reward_observation = RewardObservationWrapper(self.segments)

        # render-specific parameters
        self.render_simulation = params.traffic_graph.render
        if self.render_simulation:
            self.render_scale_factor = params.traffic_graph.render_scale_factor
            self.render_screen_size_scaled = tuple(
                self.render_scale_factor * x for x in params.traffic_graph.render_screen_size)
            self.render_screen = pygame.display.set_mode(self.render_screen_size_scaled)
            self.render_surface = pygame.Surface(params.traffic_graph.render_screen_size)
            self.render_light_mode = params.traffic_graph.render_light_mode
            if self.render_light_mode:
                self.render_surface.fill(color=(244, 244, 244))
            self.render_fps = params.traffic_graph.render_fps
            self.render_done = False
            self.render_clock = pygame.time.Clock()
            pygame.init()

        # gym-specific attributes
        self.action_space = gym.spaces.Discrete(self.num_offsets ** self.num_intersections)
        self.observation_space = self.reward_observation.observation_space
        self.reward_range = self.reward_observation.reward_range

    def _set_up_road_graph(self, params):
        self.intersections = [FourWayNoTurnsIntersection(i, params.traffic_graph.offsets, x, y,
                                                         params.traffic_graph.intersection_size) for i, (x, y) in
                              enumerate(params.traffic_graph.intersections)]

        i = 0
        # (100, "r", 0, "l", 1) is a segment of length 100 going
        #                       from right side of intersection 0
        #                       to left side of intersection 1
        for (length, from_idx, from_side, to_idx, to_side) in params.traffic_graph.segments:
            segment = Segment(i, length, self.intersections[to_idx], to_side, **params.traffic_graph)
            self.segments.append(segment)
            self.intersections[to_idx].add_entrance(to_side, segment)
            self.intersections[from_idx].add_exit(from_side, segment)
            i += 1

        for intersection in self.intersections:
            intersection.finalize()

    def _action_int_to_action_array(self, action_int):
        """Gets a string representation of the action_int in the base = number of offsets.
           Returns the numeric string left filled with num_intersections zeroes."""
        action_str = (np.base_repr(action_int, base=self.num_offsets)).zfill(self.num_intersections)

        """Returns array of ints from action_str list of strings"""
        return np.array(list(int(s) for s in list(action_str)))

    def step(self, action_int):

        # apply action into intersection(s)
        action_array = self._action_int_to_action_array(action_int)
        for intersection, action in zip(self.intersections, action_array):
            intersection.set_action(action)

        self.reward_observation.reset()

        for update in range(self.updates_per_step):

            if self.render_simulation:
                self.render()

            # update simulation
            for s in self.segments:
                s.update_first_phase()
            for s in self.segments:
                s.update_second_phase()
            for i in self.intersections:
                i.update()

            self.reward_observation.update()

        self.current_step += 1
        done = self.current_step >= self.params.traffic_graph.max_steps

        reward, observation = self.reward_observation.values()

        info = {"action_int": action_int,
                "action_array": [self.offsets_raw[act] for act in action_array],
                **self.reward_observation.info()}

        return observation, reward[0], done, info

    def reset(self):
        for s in self.segments:
            s.reset()

        self.current_step = 0

        return self.reward_observation.reset()

    def render(self, mode="human"):
        self.render_clock.tick(self.render_fps)

        for intersection in self.intersections:
            intersection.draw(self.render_surface, self.render_light_mode)

        for segment in self.segments:
            segment.draw(self.render_surface, self.render_light_mode)

        pygame.transform.scale(self.render_surface, self.render_screen_size_scaled, self.render_screen)
        pygame.display.flip()


class RewardObservationWrapper:
    def __init__(self, segments):
        # constants
        self.segments: List[Segment] = segments
        self.num_segments: int = len(segments)

        # mutable
        self.num_steps: int
        self.reward: np.ndarray
        self.observation: np.ndarray
        self._reset()

        # gym-specific attributes
        obs_low = np.zeros(shape=(2, self.num_segments), dtype=np.float)
        obs_high = np.array((np.full(shape=self.num_segments, fill_value=segments[0].max_v, dtype=np.float32),
                             np.array([s.length for s in self.segments], dtype=np.float)))
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float)
        self.reward_range = (0, float("inf"))

    def _reset(self):
        self.num_steps = 0
        self.reward = np.zeros(1, dtype=np.float32)
        self.observation = np.zeros((2, self.num_segments), dtype=np.float32)

    def _compute_reward_observation(self):
        data_per_segment = np.array([(s.total_distance(), s.mean_velocity(), s.num_cars()) for s in self.segments],
                                    dtype=np.float32).swapaxes(0, 1)

        total_distance_per_segment = data_per_segment[0, :]
        reward = np.sum(total_distance_per_segment)

        observation = data_per_segment[1:3, :]

        return reward, observation

    def reset(self):
        self._reset()

        _, observation = self._compute_reward_observation()
        return observation

    def update(self) -> None:
        self.num_steps += 1

        update_reward, update_observation = self._compute_reward_observation()

        self.reward += update_reward
        self.observation += update_observation

    def values(self):
        return self.reward, (self.observation / self.num_steps)

    def info(self) -> {}:
        total_distance = self.reward
        total_num_cars = np.sum(self.observation[1, :])

        return {"mean_speed": total_distance / total_num_cars,
                "mean_n_cars": total_num_cars / self.num_steps}