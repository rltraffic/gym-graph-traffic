INTERSECTION_SIZE = 9  # number of pixels; any odd number should work (cars will be VERY thin though)


# very customized functions, may not be perfect for every layout
def make_grid(grid_cols, grid_rows, scale_factor, segment_len=100, margin=20):
    margin = margin + segment_len
    return {"render_screen_size": (margin * (grid_cols + 1), margin * (grid_rows + 1)),
            "render_scale_factor": scale_factor,
            "intersections": [(x * (INTERSECTION_SIZE + segment_len) + margin,
                               y * (INTERSECTION_SIZE + segment_len) + margin)
                              for y in range(grid_rows) for x in range(grid_cols)],
            "segments":
                # l → r
                [(segment_len, x, "r", (x + 1) if (x + 1) % grid_cols != 0 else (x + 1) - grid_cols, "l")
                 for x in range(grid_rows * grid_cols)]
                +  # r ← l
                [(segment_len, (x + 1) if (x + 1) % grid_cols != 0 else (x + 1) - grid_cols, "l", x, "r")
                 for x in range(grid_rows * grid_cols)]
                +  # d ↑ u
                [(segment_len, x, "d", (x + grid_cols) if (x + grid_cols) < (grid_cols * grid_rows - 1) else (
                        (x + grid_cols) % (grid_cols * grid_rows)), "u") for x in range(grid_rows * grid_cols)]
                +  # u ↓ d
                [(segment_len, (x + grid_cols) if (x + grid_cols) < (grid_cols * grid_rows - 1) else (
                        (x + grid_cols) % (grid_cols * grid_rows)), "u", x, "d") for x in range(grid_rows * grid_cols)]
            }


def make_line(intersections, two_lanes, scale_factor, segment_len=100, margin=20):
    margin = margin + segment_len
    grid_rows = 1

    segments = [(segment_len, x, "r", (x + 1) if (x + 1) % intersections != 0 else (x + 1) - intersections, "l")
                for x in range(grid_rows * intersections)]
    if two_lanes:
        segments += [(segment_len, (x + 1) if (x + 1) % intersections != 0 else (x + 1) - intersections, "l", x, "r")
                     for x in range(grid_rows * intersections)]

    return {"render_screen_size": (margin * (intersections + 1), segment_len // 2),
            "render_scale_factor": scale_factor,
            "intersections": [(x * (INTERSECTION_SIZE + segment_len) + margin, margin - segment_len) for x in
                              range(intersections)],
            "segments": segments
            }
