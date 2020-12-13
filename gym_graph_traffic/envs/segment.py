from typing import Union, Tuple

import numpy as np
import pygame

from gym_graph_traffic.envs.intersection import Intersection


class Segment:

    def __init__(self, idx: int, length: int, next_intersection: Intersection, to_side, car_density: float,
                 max_v: int, prob_slow_down: float, intersection_size: int, **kwargs):

        self.idx = idx

        # graph info
        self.length = length
        self.next_intersection = next_intersection

        # cellular automata parameters
        self.car_density = car_density
        self.max_v = max_v
        self.prob_slow_down = prob_slow_down

        # cars positions and velocities
        self.p: Union[None, np.ndarray] = None  # position vector: 1 if there is a car, 0 otherwise
        self.v: Union[None, np.ndarray] = None  # velocity vector (of length equal to current number of cars on segment)

        # communication with neighbour segments about cars that cross intersections
        self.free_init_cells: int = 0
        self.new_car_at: Union[None, Tuple[int, int]] = None

        self.to_side = to_side

        # render
        self.road_width = intersection_size / 2

        # initialize cars and free init cells
        self.reset()

        # type of the intersection
        self.intersection_type = kwargs['intersection_type']

    def __str__(self) -> str:
        return str(self.idx)

    def reset(self) -> None:
        self.p = np.random.binomial(1, self.car_density, self.length)
        self.v = np.zeros(self.p.nonzero()[0].shape, dtype=np.int8)
        self._update_free_init_cells()

    def draw(self, surface, light_mode):
        pass
        (x, y, w, h) = self.next_intersection.segment_draw_coords(self.length, self.to_side)
        road_color = (192, 192, 192) if light_mode else (100, 100, 100)
        pygame.draw.rect(surface, road_color,
                         pygame.Rect(x, y, w, h))

        dy = w == self.road_width
        dx = h == self.road_width

        for cx in np.nonzero(self.p)[0]:
            cx = cx if self.to_side in "lu" else (self.length - 1 - cx)
            car_color = (162, 162, 162) if light_mode else (180, 180, 180)

            pygame.draw.rect(surface, car_color,
                             pygame.Rect((cx * dx + x), (cx * dy + y), self.road_width if dy else 1,
                                         self.road_width if dx else 1))

    def total_distance(self) -> int:
        """
        :return: Cumulative distance covered by cars during last update.
        """
        if self.v.size == 0:
            return 0
        else:
            return int(np.sum(self.v))

    def mean_velocity(self) -> float:
        """
        :return: Mean velocity of all cars during last update.
        """
        if self.v.size == 0:
            return 0.0
        else:
            return float(np.mean(self.v))

    def num_cars(self) -> int:
        """
        :return: Number of cars present at segment after last update.
        """
        return self.v.size

    def update_first_phase(self) -> None:
        """
        First phase of segment update: cellular automata step, and (sometimes) passing car to following segment.
        """

        if self.intersection_type == "FourWayNoTurnsIntersection":
            # extend p vector by free cells of following segment
            next_segment_free_cells = self.next_intersection.can_i_go(self.idx)
            if next_segment_free_cells > 0:
                self.p = np.append(self.p, np.zeros(next_segment_free_cells))
        elif self.intersection_type == "FourWayTurnsIntersection":
            # check if there is auto (1) in the last 5 cells of the segment
            if 1 in self.p[-5:]:
                # get car index from the last cell contains '1'
                car_last_index = self.p.nonzero()[0][-1]
                # Check if the car can pass the intersection, if so then save all the information into the dictionary
                # info_can_i_go =
                # {'free_cells_at_intersection': 0 or 1 or 2 or 3,
                # 'chosen_segment':[intersection.exits[chosen_direction].idx),intersection.exits[chosen_direction].to_side],
                # 'free_cells_at_segment': {car_last_index: intersection.exits[chosen_direction].free_init_cells}},
                # 'direction': "straight" or "turn right" or "turn left" }
                # Otherwise info_can_i_go = 0, so car can't go through the intersection.
                info_can_i_go = self.next_intersection.can_i_go(self.idx, car_last_index)
                if isinstance(info_can_i_go, dict):
                    if info_can_i_go.get("free_cells_at_intersection") != 0:
                        # extend p vector by free cells of intersection and following segment
                        self.p = np.append(self.p, np.zeros(info_can_i_go['free_cells_at_segment'][car_last_index] +
                                                            info_can_i_go['free_cells_at_intersection']))

        # update cellular automata if vector p contains cars
        if 1 in self.p:
            self._nagel_schreckenberg_step()

        # cut excessive cells
        self.p, next_segment_cells = np.split(self.p, [self.length])

        # if some car crossed intersection: pass car to next segment (via the intersection)
        try:
            next_sect_car_pos = next_segment_cells.tolist().index(1)
            # at any given update, only one car can cross intersection (by the rules of automata)
            self.v, next_sect_car_vel = np.split(self.v, [-1])

            if self.intersection_type == "FourWayNoTurnsIntersection":
                self.next_intersection.pass_car(self.idx, next_sect_car_pos, next_sect_car_vel[0])
            elif self.intersection_type == "FourWayTurnsIntersection":
                self.next_intersection.pass_car(self.idx, next_sect_car_pos, next_sect_car_vel,
                                                info_can_i_go['chosen_segment'][0], info_can_i_go['chosen_segment'][1],
                                                info_can_i_go['direction'])
        except:
            pass

    def update_second_phase(self) -> None:
        """
        Second phase of segment update: receiving car from preceding segment and updating info about init cells.
        """

        # receive car from preceding segment (if there is any)
        if self.new_car_at is not None:
            (pos, vel) = self.new_car_at
            self.p[pos] = 1
            self.v = np.insert(self.v, 0, vel)
            self.new_car_at = None

        # update information about init cells
        self._update_free_init_cells()

    def _nagel_schreckenberg_step(self) -> None:
        """
        Updating automata by the rules of Nagel-Schreckenberg model.
        """

        # 1. Acceleration
        self.v += 1
        self.v[self.v == self.max_v + 1] = self.max_v

        # 2. Slowing down
        cars_indices = self.p.nonzero()[0]
        cars_indices_extended = np.append(cars_indices, self.p.size)
        free_cells = cars_indices_extended[1:] - cars_indices - 1
        self.v = np.minimum(self.v, free_cells)

        # 3. Randomization
        self.v -= np.random.binomial(1, self.prob_slow_down, self.v.size)
        self.v[self.v == -1] = 0

        # 4. Car motion
        new_cars_indices = cars_indices + self.v
        self.p = np.zeros_like(self.p)
        self.p.put(new_cars_indices, 1)

    def _update_free_init_cells(self) -> None:
        """
        Updating information about init cells.
        """
        i = 0
        while i < self.max_v and self.p[i] != 1:
            i += 1
        self.free_init_cells = i
