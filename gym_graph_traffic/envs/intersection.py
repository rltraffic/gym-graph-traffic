from abc import ABC
from typing import List
import numpy as np
import random

import pygame


class Intersection(ABC):

    def __init__(self, idx):
        self.idx = idx

        self.entrances = {}
        self.exits = {}

        self.routing = None  # entrance_direction: str -> exit_direction: str
        self.dest_dict = None  # entrance_segment_idx: int -> {entrance_direction: str, exit_segment: Segment}

    def __str__(self) -> str:
        return str(self.idx)

    def add_entrance(self, dir_char, section) -> None:
        self.entrances[dir_char] = section

    def add_exit(self, dir_char, section) -> None:
        self.exits[dir_char] = section

    def finalize(self) -> None:
        raise NotImplementedError

    def can_i_go(self, segment, cars_indices) -> int:
        raise NotImplementedError

    def pass_car(self, from_idx, car_position, car_velocity, next_segment, direction) -> None:
        raise NotImplementedError

    def segment_draw_coords(self, length, to_side):
        raise NotImplementedError

class FourWayNoTurnsIntersection(Intersection):

    def __init__(self, idx, red_durations: List[int], x, y, intersection_size, max_v: int, prob_slow_down: float):
        super().__init__(idx)

        self.red_durations = red_durations
        self.routing = {"u": "d",
                        "d": "u",
                        "l": "r",
                        "r": "l"}

        self.updates_until_state_change = -1
        self.state = None
        self.update()
        self.x = x
        self.y = y
        self.intersection_size = intersection_size

    def __str__(self) -> str:
        return str(self.idx)

    def finalize(self) -> None:
        self.dest_dict = {}

        for dir_char, segment in self.entrances.items():
            destination_segment = self.exits[self.routing[dir_char]]
            self.dest_dict[segment.idx] = (dir_char, destination_segment)

    def set_action(self, action) -> None:
        self.updates_until_state_change = self.red_durations[action]
        if self.updates_until_state_change == 0:
            self.state = "lr"
        else:
            self.state = "ud"

    def draw(self, surface, light_mode):
        red = (253, 65, 30)
        red_width = 1
        road_color = (192, 192, 192) if light_mode else (100, 100, 100)
        pygame.draw.rect(surface, road_color,
                         pygame.Rect(self.x, self.y, self.intersection_size, self.intersection_size))
        if self.state is "lr":
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y, self.intersection_size / 2, red_width))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x + self.intersection_size / 2 + 1, self.y + self.intersection_size - 1,
                                         self.intersection_size / 2,
                                         red_width))
        else:
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y + self.intersection_size / 2 + 1, red_width,
                                         self.intersection_size / 2))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x + self.intersection_size - 1, self.y, red_width,
                                         self.intersection_size / 2))

    def segment_draw_coords(self, d, to_side):
        half_intersection_size = self.intersection_size / 2

        to_direction = {
            'l': [self.x - d, self.y + half_intersection_size + 1, d, half_intersection_size],
            'r': [self.x + self.intersection_size, self.y, d, half_intersection_size],
            'd': [self.x + half_intersection_size + 1, self.y + self.intersection_size, half_intersection_size, d],
            'u': [self.x, self.y - d, half_intersection_size, d]
        }

        return to_direction[to_side]

    def update(self) -> None:
        if self.state is "ud":
            self.updates_until_state_change -= 1
            if self.updates_until_state_change == 0:
                self.state = "lr"

    def can_i_go(self, from_idx) -> int:
        (source, dest) = self.dest_dict.get(from_idx, (None, None))
        if source in self.state and dest is not None:
            return dest.free_init_cells
        return 0

    def pass_car(self, from_idx, car_position, car_velocity) -> None:
        (_, dest) = self.dest_dict[from_idx]
        dest.new_car_at = (car_position, car_velocity)


class FourWayTurnsIntersection(Intersection):

    def __init__(self, idx, red_durations: List[int], x, y, intersection_size, max_v: int, prob_slow_down: float):
        super().__init__(idx)

        self.red_durations = red_durations
        self.routing = {"u": ["d", "l", "r"],
                        "d": ["u", "l", "r"],
                        "l": ["r", "u", "d"],
                        "r": ["l", "u", "d"]
                        }

        self.updates_until_state_change = -1
        self.state = None
        self.update()
        self.x = x
        self.y = y
        self.intersection_size = intersection_size

        # create matrix of cells at the intersection,
        # each intersection has 2 cells in two directions ( vertical and horizontal)
        self.cells_at_the_intersection = np.zeros((2, 2), dtype=int)

        # communication with neighbour segments about cars that are on intersection
        self.new_car_at_intersection = []
        # velocity vector
        self.v = np.zeros(0, dtype=np.int8)

        # cellular automata parameters
        self.max_v = max_v
        self.prob_slow_down = prob_slow_down

    def __str__(self) -> str:
        return str(self.idx)

    def finalize(self) -> None:
        self.dest_dict = {}
        for dir_char, segment in self.entrances.items():
            destination_segment = []
            for routing_exit_char in self.routing[dir_char]:
                destination_segment.append(self.exits[routing_exit_char])
            self.dest_dict[segment.idx] = (dir_char, destination_segment)

    def set_action(self, action) -> None:
        self.updates_until_state_change = self.red_durations[action]
        if self.updates_until_state_change == 0:
            self.state = "lr"
        else:
            self.state = "ud"

    def draw(self, surface, light_mode):
        red = (253, 65, 30)
        red_width = 1
        road_color = (192, 192, 192) if light_mode else (100, 100, 100)
        pygame.draw.rect(surface, road_color,
                         pygame.Rect(self.x, self.y, self.intersection_size, self.intersection_size))

        if self.state is "lr":
            # Check if the cells with the intersection have cars (they are equal to "1"), if so, draw a car at the intersection
            for i, check_cell in enumerate(self.cells_at_the_intersection):
                for car_in_cell in np.nonzero(check_cell)[0]:
                    if car_in_cell is not None:
                        car_color = (162, 162, 162) if light_mode else (180, 180, 180)
                        pygame.draw.rect(surface, car_color,
                                         pygame.Rect((self.x + self.intersection_size / 2 + 4 * car_in_cell - 2,
                                                      self.y + (self.intersection_size / 2 * i + 0.5), 1,
                                                      self.intersection_size / 2 - 0.5)))

            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y, self.intersection_size / 2, red_width))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x + self.intersection_size / 2 + 1, self.y + self.intersection_size - 1,
                                         self.intersection_size / 2,
                                         red_width))

        else:
            # Check if the cells with the intersection have cars (they are equal to "1"), if so, draw a car at the intersection
            for i, check_cell in enumerate(self.cells_at_the_intersection):
                for car_in_cell in np.nonzero(check_cell)[0]:
                    if car_in_cell is not None:
                        car_color = (162, 162, 162) if light_mode else (180, 180, 180)
                        pygame.draw.rect(surface, car_color,
                                         pygame.Rect((self.x + (self.intersection_size / 2 * car_in_cell + 0.5),
                                                      self.y + self.intersection_size / 2 + 4 * i - 2,
                                                      self.intersection_size / 2 - 0.5, 1)))

            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y + self.intersection_size / 2 + 1, red_width,
                                         self.intersection_size / 2))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x + self.intersection_size - 1, self.y, red_width,
                                         self.intersection_size / 2))

    def segment_draw_coords(self, d, to_side):
        half_intersection_size = self.intersection_size / 2
        to_direction = {
            'l': [self.x - d, self.y + half_intersection_size + 1, d, half_intersection_size],
            'r': [self.x + self.intersection_size, self.y, d, half_intersection_size],
            'd': [self.x + half_intersection_size + 1, self.y + self.intersection_size, half_intersection_size, d],
            'u': [self.x, self.y - d, half_intersection_size, d]
        }
        return to_direction[to_side]

    def update(self) -> None:
        if self.state is "ud":
            self.updates_until_state_change -= 1
            if self.updates_until_state_change == 0:
                self.state = "lr"

    def can_i_go(self, from_idx, cars_indices):
        source = self.dest_dict[from_idx][0]

        if source in self.state:

            """
            Check first if the intersection was left by cars from the previous
            state which were driving straight or turning left, e.g.
            If previous state was "ud" and now state is "lr" and 
            if there is a car in the cell:
            ↓[1 0]
            ↓[0 0]
            and this car is going to straight, it has priority at the
            intersection and must first leave the intersection in order
            for a car to enter from the left. The same will happen if 
            the car in this cell wants to turn left, e.g.
            ↓[1 0]      
            ↓[0 0]
              ---->   
            Otherwise, when a car from the cell wants to turn left and
            comes from the same state that is on, e.g.
            state is "lr", car comes to intersection from right
            <-----
            ↓[1 0]
            ↓[0 0]
            it must stop and give way to cars on the left side of the intersection.
            """

            # This variable stores information from which side of the intersection
            # the car entered and which direction in the example above
            side_previous_segment = ""
            direction_previous_segment = ""

            # If car can go by intersection, save all information about it
            # in directory info_can_i_go{}
            info_can_i_go = {
                'free_cells_at_intersection': None,
                'chosen_segment': [],
                'free_cells_at_segment': {},
                'direction': ""
            }

            if source is "r":
                # car can go if two cells in its direction is free
                # [0 0]     [0 0]    [0 0]
                # [0 0]  or [1 0] or [0 1]

                if np.count_nonzero(self.cells_at_the_intersection[0]) == 0:
                    # if the car is in a cell on the left,
                    # [0 0]
                    # [0 1]
                    # check if this car is a previous state or not
                    if self.cells_at_the_intersection[1][1] == 1:
                        for car in self.new_car_at_intersection:
                            side_previous_segment = self.dest_dict[car[2]][0]
                            direction_previous_segment = car[5]

                        # If the car in this cell is from the previous state and has a direction other
                        # than "turn right", the check variable is set to False. Otherwise, the variable
                        # check is True, so you car can go over the intersection.
                        check = None

                        if side_previous_segment == "d" and direction_previous_segment == "turn left":
                            check = False
                        elif side_previous_segment == "d" and direction_previous_segment == "straight":
                            check = False
                        else:
                            check = True

                    if self.cells_at_the_intersection[1][1] == 0 or (
                            self.cells_at_the_intersection[1][1] == 1 and check == True):

                        # choose direction for  car in last cell of segment
                        chosen_direction = random.choices(('l', 'u', 'd'), weights=(0.5, 0.25, 0.25), k=1)[0]
                        # determine the number of cells at the intersection
                        if chosen_direction == 'l':
                            # straight
                            info_can_i_go['free_cells_at_intersection'] = 2
                            info_can_i_go['direction'] = "straight"
                        elif chosen_direction == 'u':
                            # turn right
                            info_can_i_go['free_cells_at_intersection'] = 1
                            info_can_i_go['direction'] = "turn right"
                        else:
                            # turn left
                            info_can_i_go['direction'] = "turn left"
                            # check if there is a car in a cell [1][0] if not car can go
                            if self.cells_at_the_intersection[1][0] == 0:
                                # check if there are cars in the opposite segment
                                if np.count_nonzero(self.entrances['l'].p) != 0:
                                    # check if there is a car the last five cells in the opposite segment
                                    if (np.where(self.entrances['l'].p == 1)[0][-1]) < (self.entrances['l'].length - 5):
                                        info_can_i_go['free_cells_at_intersection'] = 3
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: self.exits[chosen_direction].free_init_cells}
                                    else:
                                        info_can_i_go['free_cells_at_intersection'] = 2
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: 0}
                                else:
                                    # nothing is coming from the opposite direction
                                    info_can_i_go['free_cells_at_intersection'] = 3
                                    info_can_i_go['free_cells_at_segment'] = {
                                        cars_indices: self.exits[chosen_direction].free_init_cells}
                            else:
                                info_can_i_go['free_cells_at_intersection'] = 2
                                # there is a car on the left-turn trajectory, so this is equal to zero
                                info_can_i_go['free_cells_at_segment'] = {
                                    cars_indices: 0}

                        # save id and side chosen segment
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].idx)
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].to_side)

                    if chosen_direction != 'd':
                        # check free cells in chosen segment
                        info_can_i_go['free_cells_at_segment'] = {
                            cars_indices: self.exits[chosen_direction].free_init_cells}

                    return info_can_i_go

            elif source is "l":
                # car can go if two cells in its direction is free
                # [0 0]     [0 1]
                # [0 0]  or [0 0]

                if np.count_nonzero(self.cells_at_the_intersection[1]) == 0:
                    # if the car is in a cell on the left,
                    # [1 0]
                    # [0 0]
                    # check if this car is a previous state or not
                    if self.cells_at_the_intersection[0][0] == 1:
                        for car in self.new_car_at_intersection:
                            side_previous_segment = self.dest_dict[car[2]][0]
                            direction_previous_segment = car[5]

                        check = None

                        if side_previous_segment == "u" and direction_previous_segment == "turn left":
                            check = False
                        elif side_previous_segment == "u" and direction_previous_segment == "straight":
                            check = False
                        else:
                            check = True

                    if self.cells_at_the_intersection[0][0] == 0 or (
                            self.cells_at_the_intersection[0][0] == 1 and check == True):

                        # choose direction for  car in last cell of segment
                        chosen_direction = random.choices(('r', 'd', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
                        # determine the number of cells at the intersection
                        if chosen_direction == 'r':
                            # straight
                            info_can_i_go['free_cells_at_intersection'] = 2
                            info_can_i_go['direction'] = "straight"
                        elif chosen_direction == 'd':
                            # turn right
                            info_can_i_go['free_cells_at_intersection'] = 1
                            info_can_i_go['direction'] = "turn right"
                        else:
                            # turn left
                            info_can_i_go['direction'] = "turn left"
                            # check if there is a car in a cell [0][1]
                            if self.cells_at_the_intersection[0][1] == 0:
                                # check if there are cars in the opposite segment
                                if np.count_nonzero(self.entrances['r'].p) != 0:
                                    # check if there is a car the last five cells in the opposite segment
                                    if (np.where(self.entrances['r'].p == 1)[0][-1]) < (self.entrances['r'].length - 5):
                                        info_can_i_go['free_cells_at_intersection'] = 3
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: self.exits[chosen_direction].free_init_cells}
                                    else:
                                        info_can_i_go['free_cells_at_intersection'] = 2
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: 0}
                                else:
                                    # nothing is coming from the opposite direction
                                    info_can_i_go['free_cells_at_intersection'] = 3
                                    info_can_i_go['free_cells_at_segment'] = {
                                        cars_indices: self.exits[chosen_direction].free_init_cells}
                            else:
                                info_can_i_go['free_cells_at_intersection'] = 2
                                # there is a car on the left-turn trajectory, so this is equal to zero
                                info_can_i_go['free_cells_at_segment'] = {
                                    cars_indices: 0}

                        # save id and side chosen segment
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].idx)
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].to_side)

                        if chosen_direction != 'u':
                            # check free cells in chosen segment
                            info_can_i_go['free_cells_at_segment'] = {
                                cars_indices: self.exits[chosen_direction].free_init_cells}

                        return info_can_i_go

            elif source is "d":
                # car can go if two cells in its direction is free
                # [0 0]     [0 0]
                # [0 0]  or [1 0]
                if self.cells_at_the_intersection[0][1] == 0 and self.cells_at_the_intersection[1][1] == 0:
                    # if the car is in a cell on the left,
                    # [0 0]
                    # [1 0]
                    # check if this car is a previous state or not
                    if self.cells_at_the_intersection[1][0] == 1:
                        for car in self.new_car_at_intersection:
                            side_previous_segment = self.dest_dict[car[2]][0]
                            direction_previous_segment = car[5]

                        check = None

                        if side_previous_segment == "l" and direction_previous_segment == "turn left":
                            check = False
                        elif side_previous_segment == "l" and direction_previous_segment == "straight":
                            check = False
                        else:
                            check = True

                    if self.cells_at_the_intersection[1][0] == 0 or (
                            self.cells_at_the_intersection[1][0] == 1 and check == True):

                        # choose direction for  car in last cell of segment
                        chosen_direction = random.choices(('u', 'r', 'l'), weights=(0.5, 0.25, 0.25), k=1)[0]
                        # determine the number of cells at the intersection
                        if chosen_direction == 'u':
                            # straight
                            info_can_i_go['free_cells_at_intersection'] = 2
                            info_can_i_go['direction'] = "straight"
                        elif chosen_direction == 'r':
                            # turn right
                            info_can_i_go['free_cells_at_intersection'] = 1
                            info_can_i_go['direction'] = "turn right"
                        else:
                            # turn left
                            info_can_i_go['direction'] = "turn left"
                            # check if there is a car in a cell [0][0]
                            if self.cells_at_the_intersection[0][0] == 0:
                                # check if there are cars in the opposite segment
                                if np.count_nonzero(self.entrances['u'].p) != 0:
                                    # check if there is a car the last five cells in the opposite segment
                                    if (np.where(self.entrances['u'].p == 1)[0][-1]) < (self.entrances['u'].length - 5):
                                        info_can_i_go['free_cells_at_intersection'] = 3
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: self.exits[chosen_direction].free_init_cells}
                                    else:
                                        info_can_i_go['free_cells_at_intersection'] = 2
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: 0}
                                else:
                                    # nothing is coming from the opposite direction
                                    info_can_i_go['free_cells_at_intersection'] = 3
                                    info_can_i_go['free_cells_at_segment'] = {
                                        cars_indices: self.exits[chosen_direction].free_init_cells}
                            else:
                                info_can_i_go['free_cells_at_intersection'] = 2
                                # there is a car on the left-turn trajectory, so this is equal to zero
                                info_can_i_go['free_cells_at_segment'] = {
                                    cars_indices: 0}

                        # save id and side chosen segment
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].idx)
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].to_side)

                        if chosen_direction != 'l':
                            # check free cells in chosen segment
                            info_can_i_go['free_cells_at_segment'] = {
                                cars_indices: self.exits[chosen_direction].free_init_cells}

                        return info_can_i_go
            else:
                # car can go if two cells in its direction is free
                # [0 0]     [0 0]
                # [0 0]  or [0 1]
                if self.cells_at_the_intersection[0][0] == 0 and self.cells_at_the_intersection[1][0] == 0:
                    # if the car is in a cell on the left,
                    # [0 1]
                    # [0 0]
                    # check if this car is a previous state or not
                    if self.cells_at_the_intersection[0][1] == 1:
                        for car in self.new_car_at_intersection:
                            side_previous_segment = self.dest_dict[car[2]][0]
                            direction_previous_segment = car[5]

                        check = None

                        if side_previous_segment == "r" and direction_previous_segment == "turn left":
                            check = False
                        elif side_previous_segment == "r" and direction_previous_segment == "straight":
                            check = False
                        else:
                            check = True

                    if self.cells_at_the_intersection[0][1] == 0 or (
                            self.cells_at_the_intersection[0][1] == 1 and check == True):

                        chosen_direction = random.choices(('d', 'l', 'r'), weights=(0.5, 0.25, 0.25), k=1)[0]
                        # determine the number of cells at the intersection
                        if chosen_direction == 'd':
                            # straight
                            info_can_i_go['free_cells_at_intersection'] = 2
                            info_can_i_go['direction'] = "straight"
                        elif chosen_direction == 'l':
                            # turn right
                            info_can_i_go['free_cells_at_intersection'] = 1
                            info_can_i_go['direction'] = "turn right"
                        else:
                            # turn left
                            info_can_i_go['direction'] = "turn left"
                            # check if there is a car in a cell [1][1]
                            if self.cells_at_the_intersection[1][1] == 0:
                                # check if there are cars in the opposite segment
                                if np.count_nonzero(self.entrances['d'].p) != 0:
                                    # check if there is a car the last five cells in the opposite segment
                                    if (np.where(self.entrances['d'].p == 1)[0][-1]) < (self.entrances['d'].length - 5):
                                        info_can_i_go['free_cells_at_intersection'] = 3
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: self.exits[chosen_direction].free_init_cells}
                                    else:
                                        info_can_i_go['free_cells_at_intersection'] = 2
                                        info_can_i_go['free_cells_at_segment'] = {
                                            cars_indices: 0}
                                else:
                                    # nothing is coming from the opposite direction
                                    info_can_i_go['free_cells_at_intersection'] = 3
                                    info_can_i_go['free_cells_at_segment'] = {
                                        cars_indices: self.exits[chosen_direction].free_init_cells}
                            else:
                                info_can_i_go['free_cells_at_intersection'] = 2
                                # there is a car on the left-turn trajectory, so this is equal to zero
                                info_can_i_go['free_cells_at_segment'] = {
                                    cars_indices: 0}

                        # save id and side chosen segment
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].idx)
                        info_can_i_go['chosen_segment'].append(self.exits[chosen_direction].to_side)

                        if chosen_direction != 'r':
                            # check free cells in chosen segment
                            info_can_i_go['free_cells_at_segment'] = {
                                cars_indices: self.exits[chosen_direction].free_init_cells}

                        return info_can_i_go
        return 0

    def pass_car(self, from_idx, car_position, car_velocity, id_next_segment, side_next_segment, direction) -> None:

        if direction == "straight":
            # car stays at intersection
            if car_position == 0 or car_position == 1:
                self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position)
                self.new_car_at_intersection.append(
                    [car_position, car_velocity, from_idx, side_next_segment, id_next_segment, direction])
            else:
                # car leaves the intersection
                for i in self.dest_dict[from_idx][1]:
                    if i.idx == id_next_segment:
                        i.new_car_at = (car_position - 2, car_velocity)
        elif direction == "turn right":
            # car stays at intersection
            if car_position == 0:
                self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position)
                self.new_car_at_intersection.append(
                    [car_position, car_velocity, from_idx, side_next_segment, id_next_segment, direction])
            else:
                # car leaves the intersection
                for i in self.dest_dict[from_idx][1]:
                    if i.idx == id_next_segment:
                        i.new_car_at = (car_position - 1, car_velocity)
        else:  # turn left
            # car stays at intersection
            if car_position == 0 or car_position == 1:
                self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position)
                self.new_car_at_intersection.append(
                    [car_position, car_velocity, from_idx, side_next_segment, id_next_segment, direction])
            elif car_position == 2:
                # car stays at intersection
                self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position)
                self.new_car_at_intersection.append(
                    [car_position, car_velocity, from_idx, side_next_segment, id_next_segment, direction])
            else:
                # car leaves the intersection
                for i in self.dest_dict[from_idx][1]:
                    if i.idx == id_next_segment:
                        i.new_car_at = (car_position - 3, car_velocity)

    def modify_the_cells_at_the_intersection(self, to_side, cell_with_a_car):
        # put the car in the cell at the intersection
        if to_side == 'u':
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(0, 1)
            elif cell_with_a_car == 1:
                self.cells_at_the_intersection.put(2, 1)
            else:
                self.cells_at_the_intersection.put(3, 1)
        elif to_side == 'd':
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(3, 1)
            elif cell_with_a_car == 1:
                self.cells_at_the_intersection.put(1, 1)
            else:
                self.cells_at_the_intersection.put(0, 1)
        elif to_side == 'l':
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(2, 1)
            elif cell_with_a_car == 1:
                self.cells_at_the_intersection.put(3, 1)
            else:
                self.cells_at_the_intersection.put(1, 1)
        else:
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(1, 1)
            elif cell_with_a_car == 1:
                self.cells_at_the_intersection.put(0, 1)
            else:
                self.cells_at_the_intersection.put(2, 1)

    def update_first_phase(self) -> None:
        """
        First phase of intersection update: cellular automata step, and (sometimes) passing car to following segment.
        """

        if np.count_nonzero(self.cells_at_the_intersection) != 0:

            # auxiliary array for removing elements from new_car_at_intersection
            delete_elements = []

            # Checking that there are two cars at the intersection that occupy the second cells (1) at the
            # intersection and want to turn left - this prevents traffic jams in the sections.
            there_are_two_cars_want_to_turn_left = None
            if len(self.new_car_at_intersection) == 2:
                if self.new_car_at_intersection[0][0] == 1 \
                        and self.new_car_at_intersection[1][0] == 1\
                        and self.new_car_at_intersection[0][5] == "turn left" \
                        and self.new_car_at_intersection[1][5] == "turn left":
                    there_are_two_cars_want_to_turn_left = True

            # Car is an array containing the car's position at the
            # intersection (0 or 1 or 2), the speed of the car, id the previous
            # segment, the side of next segment, id the next segment, direction move.
            # car = [car_position, car_velocity, from_idx, side_next_segment, id_next_segment, direction]
            # direction: straight, turn right or turn left
            for car in self.new_car_at_intersection:

                # auxiliary variable storing information from which side
                # of the intersection the next segment is located
                side_exit_from_intersection = ""
                if car[3] == "l":
                    side_exit_from_intersection = "r"
                elif car[3] == "d":
                    side_exit_from_intersection = "u"
                elif car[3] == "u":
                    side_exit_from_intersection = "d"
                else:
                    side_exit_from_intersection = "l"

                # if the first cell is occupied and car go straight
                if car[5] == "straight":
                    # e.g. [   ]
                    #  --> [1 0] -->
                    # e.g. [1  ]  ↓
                    #      [0  ]  ↓
                    pos = np.zeros((2 - car[0] + self.exits[side_exit_from_intersection].free_init_cells),
                                   dtype=np.int8)
                    # put car in the vector
                    pos[0] = 1
                    (pos, vel) = self._nagel_schreckenberg_step(pos, car[1])
                    # split cells
                    pos, next_segment_cells = np.split(pos, [2 - car[0]])

                    # if the car stays at the intersection
                    if (np.count_nonzero(next_segment_cells)) == 0:
                        # update velocity car at the intersection
                        car[1] = vel

                        # car changes its position at the intersection and occupies second cell
                        if car[0] == 0 and pos.tolist().index(1) == 1:
                            # update position car at the intersection
                            car[0] = 1
                            if car[3] == 'l':
                                self.cells_at_the_intersection.put(2, 0)
                                self.cells_at_the_intersection.put(3, 1)
                            elif car[3] == 'r':
                                self.cells_at_the_intersection.put(1, 0)
                                self.cells_at_the_intersection.put(0, 1)
                            elif car[3] == 'd':
                                self.cells_at_the_intersection.put(3, 0)
                                self.cells_at_the_intersection.put(1, 1)
                            else:
                                self.cells_at_the_intersection.put(0, 0)
                                self.cells_at_the_intersection.put(2, 1)

                    else:
                        # if the car leaves the intersection
                        # update position and velocity car in new segment
                        self.exits[side_exit_from_intersection].new_car_at = (
                            next_segment_cells.tolist().index(1), vel[0])

                        # clean the cell at the intersection
                        if car[3] == 'l':
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(2, 0)
                            else:
                                self.cells_at_the_intersection.put(3, 0)
                        elif car[3] == 'r':
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(1, 0)
                            else:
                                self.cells_at_the_intersection.put(0, 0)
                        elif car[3] == 'd':
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(3, 0)
                            else:
                                self.cells_at_the_intersection.put(1, 0)
                        else:
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(0, 0)
                            else:
                                self.cells_at_the_intersection.put(2, 0)

                        # If the car leaves the intersection, add it to a temporary
                        # array that will allow you to remove it from the array new_car_at_intersection[]
                        delete_elements.append(car)

                # if the first cell is occupied and car turn right
                elif car[5] == "turn right":
                    # e.g. [  1] ^-
                    #      [   ]
                    # e.g. [   ]
                    #      [1  ] --v
                    pos = np.zeros((1 + self.exits[side_exit_from_intersection].free_init_cells), dtype=np.int8)
                    # put car in the vector
                    pos[0] = 1
                    (pos, vel) = self._nagel_schreckenberg_step(pos, car[1])
                    # split cells - 1 cell at intersection
                    pos, next_segment_cells = np.split(pos, [1])

                    # if the car stays at the intersection
                    if (np.count_nonzero(next_segment_cells)) == 0:
                        # update velocity car at the intersection
                        car[1] = vel
                    else:
                        # if the car leaves the intersection
                        # add car to new segment
                        self.exits[side_exit_from_intersection].new_car_at = (
                            next_segment_cells.tolist().index(1), vel[0])

                        # clean the cell at the intersection
                        if car[3] == 'l':
                            self.cells_at_the_intersection.put(3, 0)
                        elif car[3] == 'r':
                            self.cells_at_the_intersection.put(0, 0)
                        elif car[3] == 'd':
                            self.cells_at_the_intersection.put(1, 0)
                        else:
                            self.cells_at_the_intersection.put(2, 0)

                        # If the car leaves the intersection, add it to a temporary
                        # array that will allow you to remove it from the array new_car_at_intersection[]
                        delete_elements.append(car)

                # if the car turns left
                else:

                    # If a car is at an intersection and the traffic light (state) has changed
                    # or the car is just leaving the intersection (car[0] = 2) or at the intersection
                    # there are two cars that occupy the second cells and want to turn left
                    # (special condition to limit the formation of traffic jams) it has priority
                    # and can cross the intersection first --> priority = True
                    # If a car is at an intersection and wants to turn left,
                    # and cars are coming in the opposite direction, it must
                    # give way to them. --> priority = False

                    priority = None

                    if car[3] in self.state or car[0] == 2 or there_are_two_cars_want_to_turn_left == True:
                        priority = True
                    else:
                        priority = False

                    side_previous_segment = self.dest_dict[car[2]][0]

                    (free_cells_at_intersection,
                     free_cells_at_next_segment) = self.check_can_i_go_to_the_next_segment_from_intersection(
                        car[0], side_previous_segment, priority)

                    pos = np.zeros((free_cells_at_intersection + free_cells_at_next_segment), dtype=np.int8)
                    # put car in the vector
                    pos[0] = 1
                    (pos, vel) = self._nagel_schreckenberg_step(pos, car[1])

                    if free_cells_at_intersection == 3:
                        # split cells - 3 cells at intersection e.g.
                        # [  0]
                        # [0,0]
                        pos, next_segment_cells = np.split(pos, [3])
                    elif free_cells_at_intersection == 2:
                        # split cells - 2 cells at intersection e.g.
                        # [  0]
                        # [0,0]
                        pos, next_segment_cells = np.split(pos, [2])
                    else:
                        # split cells - 1 cells at intersection e.g.
                        # [   ]
                        # [  0]
                        pos, next_segment_cells = np.split(pos, [1])

                    # if the car stays at the intersection
                    if (np.count_nonzero(next_segment_cells)) == 0:
                        # update velocity car at the intersection
                        car[1] = vel

                        # car changes its position at the intersection and occupies second cell
                        # if car[0] = 0 or third cell if car[0] = 1
                        if pos.tolist().index(1) == 1:
                            if car[3] == 'l':
                                if car[0] == 0:
                                    self.cells_at_the_intersection.put(0, 0)
                                    self.cells_at_the_intersection.put(2, 1)
                                else:
                                    self.cells_at_the_intersection.put(2, 0)
                                    self.cells_at_the_intersection.put(3, 1)
                            elif car[3] == 'r':
                                if car[0] == 0:
                                    self.cells_at_the_intersection.put(3, 0)
                                    self.cells_at_the_intersection.put(1, 1)
                                else:
                                    self.cells_at_the_intersection.put(1, 0)
                                    self.cells_at_the_intersection.put(0, 1)
                            elif car[3] == 'd':
                                if car[0] == 0:
                                    self.cells_at_the_intersection.put(2, 0)
                                    self.cells_at_the_intersection.put(3, 1)
                                else:
                                    self.cells_at_the_intersection.put(3, 0)
                                    self.cells_at_the_intersection.put(1, 1)
                            else:
                                if car[0] == 0:
                                    self.cells_at_the_intersection.put(1, 0)
                                    self.cells_at_the_intersection.put(0, 1)
                                else:
                                    self.cells_at_the_intersection.put(0, 0)
                                    self.cells_at_the_intersection.put(2, 1)

                            # update position car at the intersection
                            if car[0] == 0:
                                # before, e.g.
                                # [  0]       [  0]
                                # [1,0] after [0,1]
                                car[0] = 1
                            else:
                                # before, e.g.
                                # [  0]       [  1]
                                # [  1] after [  0]
                                car[0] = 2

                        elif free_cells_at_intersection == 3 and pos.tolist().index(1) == 2:
                            # before, e.g.
                            # [  0]       [  1]
                            # [1,0] after [0,0]
                            # update position car at the intersection
                            car[0] = 2
                            if car[3] == 'l':
                                self.cells_at_the_intersection.put(0, 0)
                                self.cells_at_the_intersection.put(3, 1)
                            elif car[3] == 'r':
                                self.cells_at_the_intersection.put(3, 0)
                                self.cells_at_the_intersection.put(0, 1)
                            elif car[3] == 'd':
                                self.cells_at_the_intersection.put(2, 0)
                                self.cells_at_the_intersection.put(1, 1)
                            else:
                                self.cells_at_the_intersection.put(1, 0)
                                self.cells_at_the_intersection.put(2, 1)
                    else:
                        # if the car leaves the intersection
                        # update position and velocity car in new segment
                        self.exits[side_exit_from_intersection].new_car_at = (
                            next_segment_cells.tolist().index(1), vel[0])

                        # clean the cell at the intersection
                        if car[3] == 'l':
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(0, 0)
                            elif car[0] == 1:
                                self.cells_at_the_intersection.put(2, 0)
                            else:
                                self.cells_at_the_intersection.put(3, 0)
                        elif car[3] == 'r':
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(3, 0)
                            elif car[0] == 1:
                                self.cells_at_the_intersection.put(1, 0)
                            else:
                                self.cells_at_the_intersection.put(0, 0)
                        elif car[3] == 'd':
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(2, 0)
                            elif car[0] == 1:
                                self.cells_at_the_intersection.put(3, 0)
                            else:
                                self.cells_at_the_intersection.put(1, 0)
                        else:
                            if car[0] == 0:
                                self.cells_at_the_intersection.put(1, 0)
                            elif car[0] == 1:
                                self.cells_at_the_intersection.put(0, 0)
                            else:
                                self.cells_at_the_intersection.put(2, 0)

                        # If the car leaves the intersection, add it to a temporary
                        # array that will allow you to remove it from the array new_car_at_intersection[]
                        delete_elements.append(car)

            # If the car has just left the intersection, remove its information from the array new_car_at_intersection[]
            for i in delete_elements:
                self.new_car_at_intersection.remove(i)
            delete_elements.clear()

    def _nagel_schreckenberg_step(self, pos, v):
        """
        Updating automata by the rules of Nagel-Schreckenberg model.
        """

        # 1. Acceleration
        v += 1
        v[v == self.max_v + 1] = self.max_v

        # 2. Slowing down
        cars_indices = pos.nonzero()[0]
        cars_indices_extended = np.append(cars_indices, pos.size)
        free_cells = cars_indices_extended[1:] - cars_indices - 1
        v = np.minimum(v, free_cells)

        # 3. Randomization
        v -= np.random.binomial(1, self.prob_slow_down, v.size)
        v[v == -1] = 0

        # 4. Car motion
        new_cars_indices = cars_indices + v
        pos = np.zeros_like(pos)
        pos.put(new_cars_indices, 1)

        return (pos, v)

    def check_can_i_go_to_the_next_segment_from_intersection(self, car_position, side_previous_segment, priority):
        # check if something is coming from the opposite segment

        free_cells_at_intersection = 0
        free_cells_at_next_segment = 0

        if priority == True:

            if side_previous_segment == "l":
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[0][1] == 0:
                    free_cells_at_intersection = 3 - car_position
                    free_cells_at_next_segment = self.exits['u'].free_init_cells
                else:
                    if car_position == 2:
                        free_cells_at_intersection = 1
                        free_cells_at_next_segment = self.exits['u'].free_init_cells
                    else:
                        free_cells_at_intersection = 2 - car_position
                        # there is a car on the left-turn trajectory, so this is equal to zero
                        free_cells_at_next_segment = 0
            elif side_previous_segment == "r":
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[1][0] == 0:
                    free_cells_at_intersection = 3 - car_position
                    free_cells_at_next_segment = self.exits['d'].free_init_cells
                else:
                    if car_position == 2:
                        free_cells_at_intersection = 1
                        free_cells_at_next_segment = self.exits['d'].free_init_cells
                    else:
                        free_cells_at_intersection = 2 - car_position
                        # there is a car on the left-turn trajectory, so this is equal to zero
                        free_cells_at_next_segment = 0
            elif side_previous_segment == "d":
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[0][0] == 0:
                    free_cells_at_intersection = 3 - car_position
                    free_cells_at_next_segment = self.exits['l'].free_init_cells
                else:
                    if car_position == 2:
                        free_cells_at_intersection = 1
                        free_cells_at_next_segment = self.exits['l'].free_init_cells
                    else:
                        free_cells_at_intersection = 2 - car_position
                        # there is a car on the left-turn trajectory, so this is equal to zero
                        free_cells_at_next_segment = 0
            else:
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[1][1] == 0:
                    free_cells_at_intersection = 3 - car_position
                    free_cells_at_next_segment = self.exits['r'].free_init_cells
                else:
                    if car_position == 2:
                        free_cells_at_intersection = 1
                        free_cells_at_next_segment = self.exits['r'].free_init_cells
                    else:
                        free_cells_at_intersection = 2 - car_position
                        # there is a car on the left-turn trajectory, so this is equal to zero
                        free_cells_at_next_segment = 0
        else:
            if side_previous_segment == "l":
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[0][1] == 0:
                    # check that there are cars in the segment opposite of the previous segment
                    if np.count_nonzero(self.entrances['r'].p) != 0:
                        # check if there is a car the last five cells in the opposite segment
                        if (np.where(self.entrances['r'].p == 1)[0][-1]) < (self.entrances['r'].length - 5):
                            free_cells_at_intersection = 3 - car_position
                            free_cells_at_next_segment = self.exits['u'].free_init_cells
                        else:
                            free_cells_at_intersection = 2 - car_position
                            free_cells_at_next_segment = 0
                    else:
                        # nothing is coming from the opposite direction
                        free_cells_at_intersection = 3 - car_position
                        free_cells_at_next_segment = self.exits['u'].free_init_cells
                else:
                    free_cells_at_intersection = 2 - car_position
                    # there is a car on the left-turn trajectory, so this is equal to zero
                    free_cells_at_next_segment = 0

            elif side_previous_segment == "r":
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[1][0] == 0:
                    # check that there are cars in the segment opposite of the previous segment
                    if np.count_nonzero(self.entrances['l'].p) != 0:
                        # check if there is a car the last five cells in the opposite segment
                        if (np.where(self.entrances['l'].p == 1)[0][-1]) < (self.entrances['l'].length - 5):
                            free_cells_at_intersection = 3 - car_position
                            free_cells_at_next_segment = self.exits['d'].free_init_cells
                        else:
                            free_cells_at_intersection = 2 - car_position
                            free_cells_at_next_segment = 0
                    else:
                        # nothing is coming from the opposite direction
                        free_cells_at_intersection = 3 - car_position
                        free_cells_at_next_segment = self.exits['d'].free_init_cells
                else:
                    free_cells_at_intersection = 2 - car_position
                    # there is a car on the left-turn trajectory, so this is equal to zero
                    free_cells_at_next_segment = 0
            elif side_previous_segment == "d":
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[0][0] == 0:
                    # check that there are cars in the segment opposite of the previous segment
                    if np.count_nonzero(self.entrances['u'].p) != 0:
                        # check if there is a car the last five cells in the opposite segment
                        if (np.where(self.entrances['u'].p == 1)[0][-1]) < (self.entrances['u'].length - 5):
                            free_cells_at_intersection = 3 - car_position
                            free_cells_at_next_segment = self.exits['l'].free_init_cells
                        else:
                            free_cells_at_intersection = 2 - car_position
                            free_cells_at_next_segment = 0
                    else:
                        # nothing is coming from the opposite direction
                        free_cells_at_intersection = 3 - car_position
                        free_cells_at_next_segment = self.exits['l'].free_init_cells
                else:
                    free_cells_at_intersection = 2 - car_position
                    # there is a car on the left-turn trajectory, so this is equal to zero
                    free_cells_at_next_segment = 0
            else:
                # check if there is a car in the intersection cell when turning left
                if self.cells_at_the_intersection[1][1] == 0:
                    # check that there are cars in the segment opposite of the previous segment
                    if np.count_nonzero(self.entrances['d'].p) != 0:
                        # check if there is a car the last five cells in the opposite segment
                        if (np.where(self.entrances['d'].p == 1)[0][-1]) < (self.entrances['d'].length - 5):
                            free_cells_at_intersection = 3 - car_position
                            free_cells_at_next_segment = self.exits['r'].free_init_cells
                        else:
                            free_cells_at_intersection = 2 - car_position
                            free_cells_at_next_segment = 0
                    else:
                        # nothing is coming from the opposite direction
                        free_cells_at_intersection = 3 - car_position
                        free_cells_at_next_segment = self.exits['r'].free_init_cells
                else:
                    free_cells_at_intersection = 2 - car_position
                    # there is a car on the left-turn trajectory, so this is equal to zero
                    free_cells_at_next_segment = 0
        return (free_cells_at_intersection, free_cells_at_next_segment)
