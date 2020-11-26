from abc import ABC
from typing import List, Union, Tuple
import numpy as np
import random


import pygame


class Intersection(ABC):

    def __init__(self, idx):
        self.idx = idx

        self.entrances = {}
        self.exits = {}

        self.routing = None  # entrance_direction: str -> exit_direction: str
        self.dest_dict = None  # entrance_segment_idx: int -> (entrance_direction: str, exit_segment: Segment)

        """create matrix of cells at the intersection; 
        each intersection has 2 cells in two directions ( vertical and horizontal)"""
        self.cells_at_the_intersection = np.zeros((2, 2), dtype=int)  # create matrix of cells

        self.new_car_at_intersection = []
        self.free_init_cells: int = 4 - np.count_nonzero(self.cells_at_the_intersection)
        self.v = np.zeros(0, dtype=np.int8)

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

    def __init__(self, idx, red_durations: List[int], x, y, intersection_size):
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

            # sprawdzenie czy komórki ze skrzyzowaniu maja auta, czyli są rowne "1", jesli tak to narysuj auto na skrzyzowaniu
            for i, check_cell in enumerate(self.cells_at_the_intersection):
                for car_in_cell in np.nonzero(check_cell)[0]:
                    if car_in_cell is not None:
                        # change color for 162, 162, 162
                        car_color = (0, 0, 254) if light_mode else (180, 180, 180)
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

            # sprawdzenie czy komórki ze skrzyzowaniu maja auta, czyli są rowne "1", jesli tak to narysuj auto na skrzyzowaniu
            for i, check_cell in enumerate(self.cells_at_the_intersection):
                for car_in_cell in np.nonzero(check_cell)[0]:
                    if car_in_cell is not None:
                        # change color for 162, 162, 162
                        car_color = (0, 0, 254) if light_mode else (180, 180, 180)
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
                #if np.count_nonzero(self.cells_at_the_intersection) != 0:
                    #self.state = "lr"
                self.state = "lr"

    def can_i_go(self, segment, cars_indices):
        (source, dest) = self.dest_dict.get(segment.idx, (None, None))

        if source in self.state and dest is not None:

            if source is "r":
                # car can go if two cells in its direction is free and the cell on the left is also free
                # [0 0]     [0 0]
                # [0 0]  or [1 0]
                #  otherwise it can be accident when
                # [0 0]
                # [0 1]
                if np.count_nonzero(self.cells_at_the_intersection[0]) == 0 and self.cells_at_the_intersection[1][1] == 0:
                    info_free_cells = {
                        'free_cells_at_intersection': None,
                        'chosen_segment': [],
                        'free_cells_at_segment': {},
                        'direction': ""
                    }

                    # choose direction for  car in last cell of segment
                    # TODO change
                    # chosen_direction = random.choices(('l', 'u', 'd'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    chosen_direction = random.choices(('u', 'u', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    # determine the number of cells at the intersection
                    if chosen_direction == 'l':
                        # straight
                        info_free_cells['free_cells_at_intersection'] = 2
                        info_free_cells['direction'] = "straight"
                    elif chosen_direction == 'u':
                        # turn right
                        info_free_cells['free_cells_at_intersection'] = 1
                        info_free_cells['direction'] = "turn right"
                    else:
                        # turn left
                        # TODO
                        info_free_cells['free_cells_at_intersection'] = 3
                        info_free_cells['direction'] = "turn left"

                        # save chosen segment
                    info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
                    # check free cells in chosen segment
                    info_free_cells['free_cells_at_segment'] = {
                        cars_indices[-1]: self.exits[chosen_direction].free_init_cells}

                    return info_free_cells
            elif source is "l":
                if np.count_nonzero(self.cells_at_the_intersection[1]) == 0 and self.cells_at_the_intersection[0][0] == 0:
                #if not np.any(self.cells_at_the_intersection[1]):
                    info_free_cells = {
                        'free_cells_at_intersection': None,
                        'chosen_segment': [],
                        'free_cells_at_segment': {},
                        'direction': ""
                    }

                    # choose direction for  car in last cell of segment
                    # TODO change
                    # chosen_direction = random.choices(('r', 'd', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    chosen_direction = random.choices(('r', 'r', 'r'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    # determine the number of cells at the intersection
                    if chosen_direction == 'r':
                        # straight
                        info_free_cells['free_cells_at_intersection'] = 2
                        info_free_cells['direction'] = "straight"
                    elif chosen_direction == 'd':
                        # turn right
                        info_free_cells['free_cells_at_intersection'] = 1
                        info_free_cells['direction'] = "turn right"
                    else:
                        # turn left
                        info_free_cells['free_cells_at_intersection'] = 3
                        info_free_cells['direction'] = "turn left"

                    # save chosen segment
                    info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
                    # check free cells in chosen segment
                    info_free_cells['free_cells_at_segment'] = {
                        cars_indices[-1]: self.exits[chosen_direction].free_init_cells}

                    return info_free_cells
            elif source is "d":
                if self.cells_at_the_intersection[0][1] == 0 and self.cells_at_the_intersection[1][1] == 0 and self.cells_at_the_intersection[1][0] == 0:
                    info_free_cells = {
                        'free_cells_at_intersection': None,
                        'chosen_segment': [],
                        'free_cells_at_segment': {},
                        'direction': ""
                    }
                    # choose direction for  car in last cell of segment
                    # TODO change
                    # chosen_direction = random.choices(('u', 'r', 'l'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    chosen_direction = random.choices(('u', 'u', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    # determine the number of cells at the intersection
                    if chosen_direction == 'u':
                        # straight
                        info_free_cells['free_cells_at_intersection'] = 2
                        info_free_cells['direction'] = "straight"
                    elif chosen_direction == 'r':
                        # turn right
                        info_free_cells['free_cells_at_intersection'] = 1
                        info_free_cells['direction'] = "turn right"
                    else:
                        # turn left
                        info_free_cells['free_cells_at_intersection'] = 3
                        info_free_cells['direction'] = "turn left"
                    # save chosen segment
                    info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
                    # check free cells in chosen segment
                    info_free_cells['free_cells_at_segment'] = {
                        cars_indices[-1]: self.exits[chosen_direction].free_init_cells}

                    return info_free_cells
            else:
                if self.cells_at_the_intersection[0][0] == 0 and self.cells_at_the_intersection[1][0] == 0 and self.cells_at_the_intersection[1][1] == 0:
                    info_free_cells = {
                        'free_cells_at_intersection': None,
                        'chosen_segment': [],
                        'free_cells_at_segment': {},
                        'direction': ""
                    }

                    # TODO change
                    # chosen_direction = random.choices(('d', 'l', 'r'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    chosen_direction = random.choices(('d', 'd', 'd'), weights=(0.5, 0.25, 0.25), k=1)[0]
                    # determine the number of cells at the intersection
                    if chosen_direction == 'd':
                        # straight
                        info_free_cells['free_cells_at_intersection'] = 2
                        info_free_cells['direction'] = "straight"
                    elif chosen_direction == 'l':
                        # turn right
                        info_free_cells['free_cells_at_intersection'] = 1
                        info_free_cells['direction'] = "turn right"
                    else:
                        # turn left
                        info_free_cells['free_cells_at_intersection'] = 3
                        info_free_cells['direction'] = "turn left"
                    # save chosen segment
                    info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
                    # check free cells in chosen segment
                    info_free_cells['free_cells_at_segment'] = {
                        cars_indices[-1]: self.exits[chosen_direction].free_init_cells}

                    return info_free_cells

            # if source is "r":
            #     if not np.any(self.cells_at_the_intersection[0]):
            #
            #         info_free_cells = {
            #             'free_cells_at_intersection': len(self.cells_at_the_intersection[0]),
            #             'chosen_segment': [],
            #             'free_cells_at_segment': {}
            #         }
            #         segments_dict = {}
            #
            #         for i, j in enumerate(cars_indices):
            #             #choose direction for each car in last cells of segment
            #             #TODO change
            #             #chosen_direction = random.choices(('l', 'd', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             chosen_direction = random.choices(('l', 'l', 'l'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             #save chosen segment
            #             info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
            #             #check free cells in chosen segment
            #             segments_dict[j] = self.exits[chosen_direction].free_init_cells
            #             info_free_cells['free_cells_at_segment'] = segments_dict
            #
            #         return info_free_cells
            # elif source is "l":
            #     if not np.any(self.cells_at_the_intersection[1]):
            #
            #         info_free_cells = {
            #             'free_cells_at_intersection': len(self.cells_at_the_intersection[0]),
            #             'chosen_segment': [],
            #             'free_cells_at_segment': {}
            #         }
            #         segments_dict = {}
            #
            #         for i, j in enumerate(cars_indices):
            #             # choose direction for each car in last cells of segment
            #             #TODO change
            #             #chosen_direction = random.choices(('r', 'd', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             chosen_direction = random.choices(('r', 'r', 'r'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             # save chosen segment
            #             info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
            #             # check free cells in chosen segment
            #             segments_dict[j] = self.exits[chosen_direction].free_init_cells
            #             info_free_cells['free_cells_at_segment'] = segments_dict
            #
            #         """segments_dict = {}
            #         for i, j in enumerate(cars_indices):
            #             # segments_dict[j] = dest.free_init_cells
            #             chosen_direction = random.choices(('r', 'd', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             segments_dict[j] = self.exits.get(chosen_direction).free_init_cells
            #
            #         free_cells = {
            #             'free_init_cells_intersection': len(self.cells_at_the_intersection[0]),
            #             'free_init_cells': segments_dict,
            #             'chosen_segment': dest
            #         }
            #         print(segments_dict)"""
            #
            #         return info_free_cells
            # elif source is "d":
            #     if self.cells_at_the_intersection[0][1] == 0 and self.cells_at_the_intersection[1][1] == 0:
            #
            #         info_free_cells = {
            #             'free_cells_at_intersection': len(self.cells_at_the_intersection[0]),
            #             'chosen_segment': [],
            #             'free_cells_at_segment': {}
            #         }
            #         segments_dict = {}
            #
            #         for i, j in enumerate(cars_indices):
            #             # choose direction for each car in last cells of segment
            #             #TODO change
            #             #chosen_direction = random.choices(('u', 'l', 'r'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             chosen_direction = random.choices(('u', 'u', 'u'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             # save chosen segment
            #             info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
            #             # check free cells in chosen segment
            #             segments_dict[j] = self.exits[chosen_direction].free_init_cells
            #             info_free_cells['free_cells_at_segment'] = segments_dict
            #
            #         return info_free_cells
            # else:
            #     if self.cells_at_the_intersection[0][0] == 0 and self.cells_at_the_intersection[0][0] == 0:
            #
            #         info_free_cells = {
            #             'free_cells_at_intersection': len(self.cells_at_the_intersection[0]),
            #             'chosen_segment': [],
            #             'free_cells_at_segment': {}
            #         }
            #         segments_dict = {}
            #
            #         for i, j in enumerate(cars_indices):
            #             # choose direction for each car in last cells of segment
            #             #TODO change
            #             #chosen_direction = random.choices(('d', 'l', 'r'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             chosen_direction = random.choices(('d', 'd', 'd'), weights=(0.5, 0.25, 0.25), k=1)[0]
            #             # save chosen segment
            #             info_free_cells['chosen_segment'].append(self.exits[chosen_direction])
            #             # check free cells in chosen segment
            #             segments_dict[j] = self.exits[chosen_direction].free_init_cells
            #             info_free_cells['free_cells_at_segment'] = segments_dict
            #
            #         return info_free_cells

        return 0

    def pass_car(self, from_idx, car_position, car_velocity, next_segment, direction) -> None:

        # TODO change conditions
        # if l->r or r<-l or d ↑ u or u ↓ d
        if direction == "straight":
            #if car_position[0] == 0 or car_position[0] == 1:
            if car_position == 0 or car_position == 1:
                #self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position[0])
                self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position)
                # TODO CHANGE when car wants to turn left (if it turns left it is True,otherwise it is false
                #self.new_car_at_intersection.append([car_position[0], car_velocity, next_segment.to_side, next_segment, direction])
                self.new_car_at_intersection.append([car_position, car_velocity, next_segment.to_side, next_segment, direction])
                # update information about init cells
                self._update_free_init_cells()
            else:
                # TODO change car_position, because there are 3 ways (1/2/3 take the cells)
                #next_segment.new_car_at = (car_position[0] - 2, car_velocity)
                next_segment.new_car_at = (car_position - 2, car_velocity)
        elif direction == "turn right":
            #if car_position[0] == 0:
            if car_position == 0:
                #self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position[0])
                self.modify_the_cells_at_the_intersection(self.dest_dict[from_idx][0], car_position)
                #self.new_car_at_intersection.append([car_position[0], car_velocity, next_segment.to_side, next_segment, direction])
                self.new_car_at_intersection.append([car_position, car_velocity, next_segment.to_side, next_segment, direction])
                # update information about init cells
                self._update_free_init_cells()
            else:
                #next_segment.new_car_at = (car_position[0] - 1, car_velocity)
                next_segment.new_car_at = (car_position - 1, car_velocity)
        else:
        # direction == "turn left"



            print("change")
            #TODO
        # (_, dest) = self.dest_dict[from_idx]
        # dest.new_car_at = (car_position, car_velocity)
        # next_segment_free_cells.new_car_at = (car_position, car_velocity)

    def modify_the_cells_at_the_intersection(self, to_side, cell_with_a_car):
        # put the car in the cell at the intersection
        if to_side == 'u':
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(0, 1)
            else:
                self.cells_at_the_intersection.put(2, 1)
        elif to_side == 'd':
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(3, 1)
            else:
                self.cells_at_the_intersection.put(1, 1)
        elif to_side == 'l':
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(2, 1)
            else:
                self.cells_at_the_intersection.put(3, 1)
        else:
            if cell_with_a_car == 0:
                self.cells_at_the_intersection.put(1, 1)
            else:
                self.cells_at_the_intersection.put(0, 1)

    def update_first_phase(self) -> None:
        """
        First phase of segment update: cellular automata step, and (sometimes) passing car to following segment.
        """

        if np.count_nonzero(self.cells_at_the_intersection) != 0:

            if np.count_nonzero(self.cells_at_the_intersection) != len(self.new_car_at_intersection):
                print("roznica!!!!!!!")

            # auxiliary array for removing elements from new_car_at_intersection
            delete_elements = []

            # Car is an array containing the car's position at the
            # intersection (0 or 1), the speed of the car, the direction
            # it came from, the next segment where it wants to go and
            # whether the car turns left or not.
            # car = [car_position, car_velocity, from_side, next_segment, direction]
            # direction: straight, turn right or turn left
            for car in self.new_car_at_intersection:

                # if the first cell is occupied and car go straight
                if car[0] == 0 and car[4] == "straight":
                    # e.g. [   ]
                    #  --> [1 0] -->
                    # e.g. [1  ]  ↓
                    #      [0  ]  ↓
                    pos = np.zeros((2 + car[3].free_init_cells), dtype=np.int8)
                    # put car in the vector
                    pos[0] = 1
                    (pos, vel) = self._nagel_schreckenberg_step(pos, car[1])
                    # split cells - 2 cells at intersection e.g. [0,0]
                    pos, next_segment_cells = np.split(pos, [2])

                    # if the car stays at the intersection
                    if (np.count_nonzero(next_segment_cells)) == 0:
                        # update velocity car at the intersection
                        car[1] = vel

                        # car changes its position at the intersection and occupies second cell
                        if pos.tolist().index(1) == 1:
                            # update position car at the intersection
                            car[0] = 1
                            if car[2] == 'l':
                                self.cells_at_the_intersection.put(2, 0)
                                self.cells_at_the_intersection.put(3, 1)
                            elif car[2] == 'r':
                                self.cells_at_the_intersection.put(1, 0)
                                self.cells_at_the_intersection.put(0, 1)
                            elif car[2] == 'd':
                                self.cells_at_the_intersection.put(3, 0)
                                self.cells_at_the_intersection.put(1, 1)
                            else:
                                self.cells_at_the_intersection.put(0, 0)
                                self.cells_at_the_intersection.put(2, 1)

                    else:
                    # if the car leaves the intersection
                        # update position car in new segment
                        #car[3].p[next_segment_cells.tolist().index(1)] = 1
                        # update velocity car in new segment
                        #car[3].v = np.insert(self.v, 0, vel)
                        car[3].new_car_at = (next_segment_cells.tolist().index(1), vel[0])

                        # clean the cell at the intersection
                        if car[2] == 'l':
                            self.cells_at_the_intersection.put(2, 0)
                        elif car[2] == 'r':
                            self.cells_at_the_intersection.put(1, 0)
                        elif car[2] == 'd':
                            self.cells_at_the_intersection.put(3, 0)
                        else:
                            self.cells_at_the_intersection.put(0, 0)

                        # delete array contains car at the intersection
                        #del car
                        #self.new_car_at_intersection.remove(car)
                        delete_elements.append(car)

                # if the second cell is occupied
                elif car[0] == 1 and car[4] == "straight":
                    # e.g. [  1]-->
                    #      [   ]
                    # e.g. [   ]
                    #      [1  ] ↓
                    pos = np.zeros((1 + car[3].free_init_cells), dtype=np.int8)
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
                        # update position car in new segment
                        #car[3].p[next_segment_cells.tolist().index(1)] = 1
                        # update velocity car in new segment
                        #car[3].v = np.insert(self.v, 0, vel)
                        car[3].new_car_at = (next_segment_cells.tolist().index(1), vel[0])

                        # clean the cell at the intersection
                        if car[2] == 'l':
                            self.cells_at_the_intersection.put(3, 0)
                        elif car[2] == 'r':
                            self.cells_at_the_intersection.put(0, 0)
                        elif car[2] == 'd':
                            self.cells_at_the_intersection.put(1, 0)
                        else:
                            self.cells_at_the_intersection.put(2, 0)

                        # delete array contains car at the intersection
                        #del car
                        #self.new_car_at_intersection.remove(car)
                        delete_elements.append(car)

                # if the first cell is occupied and car turn right
                elif car[0] == 0 and car[4] == "turn right":
                    # TODO TURN RIGHT
                    print("turn right")

                    # e.g. [  1] ^-
                    #      [   ]
                    # e.g. [   ]
                    #      [1  ] --v
                    pos = np.zeros((1 + car[3].free_init_cells), dtype=np.int8)
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
                        car[3].new_car_at = (next_segment_cells.tolist().index(1), vel[0])

                        # clean the cell at the intersection
                        if car[2] == 'l':
                            self.cells_at_the_intersection.put(3, 0)
                        elif car[2] == 'r':
                            self.cells_at_the_intersection.put(0, 0)
                        elif car[2] == 'd':
                            self.cells_at_the_intersection.put(1, 0)
                        else:
                            self.cells_at_the_intersection.put(2, 0)

                        delete_elements.append(car)

                # if the third cell is occupied
                else:
                    # TODO change when is turn left
                    print("turn left")

            for i in delete_elements:
                self.new_car_at_intersection.remove(i)
            delete_elements.clear()

        self._update_free_init_cells()

    def _nagel_schreckenberg_step(self, pos, v):
        """
        Updating automata by the rules of Nagel-Schreckenberg model.
        """

        max_v = 5
        prob_slow_down = 0.1

        # 1. Acceleration
        v += 1
        v[v == max_v + 1] = max_v

        # 2. Slowing down
        cars_indices = pos.nonzero()[0]
        cars_indices_extended = np.append(cars_indices, pos.size)
        free_cells = cars_indices_extended[1:] - cars_indices - 1

        if len(v) > len(free_cells):
            print("ERROR IN NAGEL_SCHRECKENBERG_STEP() - intersection")

        v = np.minimum(v, free_cells)

        # 3. Randomization
        v -= np.random.binomial(1, prob_slow_down, v.size)
        v[v == -1] = 0

        # 4. Car motion
        new_cars_indices = cars_indices + v
        pos = np.zeros_like(pos)
        pos.put(new_cars_indices, 1)

        return (pos, v)

    def _update_free_init_cells(self) -> None:
        """
        Updating information about init cells.
        """
        i = 0
        # TODO
        while i < (4 - np.count_nonzero(self.cells_at_the_intersection)):
            i += 1
        self.free_init_cells = i


