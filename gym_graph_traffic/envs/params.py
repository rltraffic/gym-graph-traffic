from attrdict import AttrDict
from typing import Dict

from util import grid
from util.grid import INTERSECTION_SIZE

EPISODES = 100
STEPS_PER_EPISODE = 200
RENDER = False
LIGHT_MODE = False

SEED = 170017
UPDATE = 2.7  # seconds per traffic model update
STEP = 60  # seconds of traffic model simulation per step

RED_DURATIONS = [0, 20, 40, 60]  # possible red_durations
V_MAX = 5

PRESET = "grid_3x3"  # see PRESETS dictionary below

SEGMENT_LEN = 100

PRESETS = {
    "easy": grid.make_line(4, False, 3, segment_len=SEGMENT_LEN, margin=20),
    "grid_4x2": grid.make_grid(4, 2, 4, segment_len=SEGMENT_LEN, margin=20),
    "grid_3x3": grid.make_grid(3, 3, 3, segment_len=SEGMENT_LEN, margin=20),
    "two_roads": grid.make_line(4, True, 3, segment_len=SEGMENT_LEN, margin=20),
}

PARAMETERS = AttrDict({"run": {"episodes": EPISODES,
                               "max_steps": STEPS_PER_EPISODE,
                               "seed": SEED},
                       "traffic_graph": {"preset_name": PRESET,
                                         **PRESETS[PRESET],
                                         "max_steps": STEPS_PER_EPISODE,
                                         "updates_per_step": int(STEP / UPDATE),
                                         "init_car_density": 0.125,
                                         "max_v": 5,
                                         "prob_slow_down": 0.1,
                                         "red_durations": [int(o / UPDATE) for o in RED_DURATIONS],
                                         "red_durations_raw": RED_DURATIONS,
                                         "render": RENDER,
                                         "render_light_mode": LIGHT_MODE,
                                         "render_fps": 30,
                                         "intersection_size": INTERSECTION_SIZE
                                         }
                       })

assert all(0 <= off <= STEP for off in RED_DURATIONS)


def _flatten(dict_of_dicts: Dict, sep="__") -> Dict:
    """Warning. Not working for parameter list nested more than once. I.e. {{{}}}."""
    flattened = {}
    for key, value in dict_of_dicts.items():
        for sub_key, sub_value in value.items():
            flattened[key + sep + sub_key] = sub_value
    return flattened


PARAMETERS_FLATTENED = AttrDict(_flatten(PARAMETERS))
