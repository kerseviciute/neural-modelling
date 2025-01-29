from matplotlib.patches import Rectangle

screen_width, screen_height = 1680, 1050
table_width = screen_width - 100
table_height = int(screen_height * 0.9)

zone_width = int(table_width * 0.95)
zone_height = 150

scoring_rect = {
    "x": (screen_width - zone_width) // 2,
    "y": int(table_height * 0.2),
    "width": zone_width,
    "height": zone_height
}

# Calculate the corner properties
scoring_rect["topleft"] = (scoring_rect["x"], scoring_rect["y"])
scoring_rect["topright"] = (scoring_rect["x"] + scoring_rect["width"], scoring_rect["y"])
scoring_rect["bottomleft"] = (scoring_rect["x"], scoring_rect["y"] + scoring_rect["height"])
scoring_rect["bottomright"] = (scoring_rect["x"] + scoring_rect["width"], scoring_rect["y"] + scoring_rect["height"])

green_triangle = [
    scoring_rect["topleft"],
    scoring_rect["topright"],
    scoring_rect["bottomleft"]
]

red_triangle = [
    scoring_rect["bottomright"],
    scoring_rect["bottomleft"],
    scoring_rect["topright"]
]

screen = Rectangle((0, 0), screen_width, screen_height, alpha = 0.05, label = "Screen")
table = Rectangle(
    ((screen_width - table_width) // 2, (screen_height - table_height) // 2),
    table_width, table_height,
    alpha = 0.05,
    color = "orange",
    label = "Table"
)

green_x, green_y = zip(*green_triangle)
red_x, red_y = zip(*red_triangle)
