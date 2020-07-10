from abc import ABC
from typing import List

import pygame

class Intersection(ABC):

    def __init__(self, idx):
        self.idx = idx

        self.entrances = {}
        self.exits = {}

        self.routing = None  # entrance_direction: str -> exit_direction: str
        self.dest_dict = None  # entrance_segment_idx: int -> (entrance_direction: str, exit_segment: Segment)

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
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y, self.intersection_size, red_width))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y + self.intersection_size - 1, self.intersection_size,
                                         red_width))
        else:
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x, self.y, red_width, self.intersection_size))
            pygame.draw.rect(surface, red,
                             pygame.Rect(self.x + self.intersection_size - 1, self.y, red_width,
                                         self.intersection_size))

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
