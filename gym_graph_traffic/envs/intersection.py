from abc import ABC
from typing import List
import numpy as np

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
        self.cells_at_the_intersection = np.zeros((2,2),dtype = int) #create matrix of cells

    def __str__(self) -> str:
        return str(self.idx)

    def add_entrance(self, dir_char, section) -> None:
        self.entrances[dir_char] = section

    def add_exit(self, dir_char, section) -> None:
        self.exits[dir_char] = section

    def finalize(self) -> None:
        raise NotImplementedError

    def can_i_go(self, from_idx) -> int:
        raise NotImplementedError

    def pass_car(self, from_idx, car_position, car_velocity) -> None:
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
                        #change color for 162, 162, 162
                        car_color = (0, 0, 254) if light_mode else (180, 180, 180)
                        pygame.draw.rect(surface, car_color,
                                        pygame.Rect((self.x + self.intersection_size/2 + 4 * car_in_cell - 2, self.y + (self.intersection_size/2 * i + 0.5), 1, self.intersection_size/2 - 0.5)))

            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y, self.intersection_size/2, red_width))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x + self.intersection_size/2 + 1, self.y + self.intersection_size - 1, self.intersection_size/2,
                                         red_width))

        else:

            #sprawdzenie czy komórki ze skrzyzowaniu maja auta, czyli są rowne "1", jesli tak to narysuj auto na skrzyzowaniu
            for i, check_cell in enumerate(self.cells_at_the_intersection):
                for car_in_cell in np.nonzero(check_cell)[0]:
                    if car_in_cell is not None:
                        # change color for 162, 162, 162
                        car_color = (0, 0, 254) if light_mode else (180, 180, 180)
                        pygame.draw.rect(surface, car_color,
                                    pygame.Rect((self.x + (self.intersection_size/2 * car_in_cell + 0.5), self.y + self.intersection_size/2 + 4 * i - 2, self.intersection_size/2 - 0.5, 1)))

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
        #self.cells_at_the_intersection.put(3,1)
        if self.state is "ud":
            self.updates_until_state_change -= 1
            if self.updates_until_state_change == 0:
                self.state = "lr"

    def can_i_go(self, segment, cars_indices):
        (source, dest) = self.dest_dict.get(segment.idx, (None, None))

        if source in self.state and dest is not None:
            if source is "r":
                if not np.any(self.cells_at_the_intersection[0]):
                    free_cells = {
                        'intersection': len(self.cells_at_the_intersection[0]),
                        'segment': dest.free_init_cells,
                        'chosen_segment': dest
                    }
                    return free_cells
            elif source is "l":
                if not np.any(self.cells_at_the_intersection[1]):
                    free_cells = {
                        'intersection': len(self.cells_at_the_intersection[0]),
                        'segment': dest.free_init_cells,
                        'chosen_segment': dest
                    }
                    return free_cells
            elif source is "d":
                if self.cells_at_the_intersection[0][1] == 0 and self.cells_at_the_intersection[1][1] == 0:
                    free_cells = {
                        'intersection': len(self.cells_at_the_intersection[0]),
                        'segment': dest.free_init_cells,
                        'chosen_segment': dest
                    }
                    return free_cells
            else:
                if self.cells_at_the_intersection[0][0] == 0 and self.cells_at_the_intersection[0][0] == 0:
                    free_cells = {
                        'intersection': len(self.cells_at_the_intersection[0]),
                        'segment': dest.free_init_cells,
                        'chosen_segment': dest
                    }
                    return free_cells

        return 0


    def pass_car(self, from_idx, car_position, car_velocity) -> None:
        (_, dest) = self.dest_dict[from_idx]
        dest.new_car_at = (car_position, car_velocity)
