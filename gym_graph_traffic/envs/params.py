from attrdict import AttrDict

from util import grid
from util.grid import INTERSECTION_SIZE

# traffic lights
STEPS_PER_EPISODE = 200
SECONDS_PER_UPDATE = 2.7
STEP_LENGTH = 60  # in seconds
RED_DURATIONS = [0, 20, 40, 60]  # table of all possible red durations (see: README)

# cars' movement
MAX_SPEED = 5  # in cells per update
PROB_SLOW_DOWN = 0.1

# road network
PRESET = "easy"  # see PRESETS dictionary below
SEGMENT_LENGTH = 100  # in cells
CAR_DENSITY = 0.125

# rendering
RENDER = False
RENDER_LIGHT_MODE = True
RENDER_FPS = 30

PRESETS = {
    "easy": grid.make_line(4, False, 3, segment_len=SEGMENT_LENGTH),
    "grid_4x2": grid.make_grid(4, 2, 4, segment_len=SEGMENT_LENGTH),
    "grid_3x3": grid.make_grid(3, 3, 2, segment_len=SEGMENT_LENGTH),
    "two_roads": grid.make_line(4, True, 3, segment_len=SEGMENT_LENGTH),
}

PARAMETERS = AttrDict({"preset_name": PRESET,
                       **PRESETS[PRESET],
                       "steps_per_episode": STEPS_PER_EPISODE,
                       "updates_per_step": int(STEP_LENGTH / SECONDS_PER_UPDATE),
                       "car_density": CAR_DENSITY,
                       "max_v": MAX_SPEED,
                       "prob_slow_down": PROB_SLOW_DOWN,
                       "red_durations": [int(o / SECONDS_PER_UPDATE) for o in RED_DURATIONS],
                       "red_durations_raw": RED_DURATIONS,
                       "render": RENDER,
                       "render_light_mode": RENDER_LIGHT_MODE,
                       "render_fps": RENDER_FPS,
                       "intersection_size": INTERSECTION_SIZE})

assert all(0 <= off <= STEP_LENGTH for off in RED_DURATIONS)
