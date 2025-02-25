import pandas as pd
import pygame
import math
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Initialize Pygame
pygame.init()

# Screen settings
# Full screen mode
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
pygame.display.set_caption("Beer Pint Game")

# Constants
TABLE_WIDTH = SCREEN_WIDTH - 100
TABLE_HEIGHT = int(SCREEN_HEIGHT * 0.9)
FREE_ZONE_RADIUS = 110
START_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - FREE_ZONE_RADIUS - 10)

# Colors
WHITE = (255, 255, 255)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)
LIGHT_RED = (255, 182, 193)
DARK_BROWN = (120, 66, 40)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
GREEN_LAMP = (0, 255, 0)
RED_LAMP = (255, 0, 0)

# Game settings
# Parameter to control by how much to decrease the friction while drinking beer
friction_decrease = 0.003   
BASE_FRICTION = 0.99 - (3 * friction_decrease)

beer_alpha = 1
beer_alpha_decrease = 0.1

ZONE_WIDTH = int(TABLE_WIDTH * 0.95)
ZONE_HEIGHT = 150

# Rectangles and triangles
SCORING_RECT = pygame.Rect(
    (SCREEN_WIDTH - ZONE_WIDTH) // 2,
    int(TABLE_HEIGHT * 0.2),
    ZONE_WIDTH,
    ZONE_HEIGHT,
)

TABLE_RECT = pygame.Rect((SCREEN_WIDTH - TABLE_WIDTH) // 2, (SCREEN_HEIGHT - TABLE_HEIGHT) // 2, TABLE_WIDTH, TABLE_HEIGHT)

GREEN_TRIANGLE = [
    SCORING_RECT.topleft,
    SCORING_RECT.topright,
    SCORING_RECT.bottomleft
]
RED_TRIANGLE = [
    SCORING_RECT.bottomright,
    SCORING_RECT.bottomleft,
    SCORING_RECT.topright
]

# Game variables
pint_pos = list(START_POS)
pint_velocity = [0, 0]
pint_radius = 15
friction = BASE_FRICTION
launched = False
stopped = False
waiting_for_mouse = True
perturbation_active = False
score = 0
feedback_mode = False
feedback_type = None
trajectory = []
perturbation_force=0
force_increment=0.2
end_pos = [0, 0]
current_block = 1
show_info=0
trial_positions = []
last_trajectory=[]

previous_win = False
previous_pos = None
previous_trajectory = []

gradual_perturbation = False

# Font setup
font = pygame.font.SysFont(None, 36)

trial_counter = 0
results = []

noise_mean = 0
noise_std = 0
noise_active = False
noise_instance = 0

ideal_velocity = [0, 0]
ideal_trajectory = []
ideal_pos = list(START_POS)

# BUILD FIELD
def draw_playfield(mask_pint=False):
    """Draw the game playfield."""
    screen.fill(WHITE)

    # Draw the table
    pygame.draw.rect(screen, DARK_BROWN, TABLE_RECT)

    # Draw free movement zone
    pygame.draw.circle(screen, LIGHT_BLUE, START_POS, FREE_ZONE_RADIUS)
    pygame.draw.circle(screen, BLACK, START_POS, FREE_ZONE_RADIUS, 3)

    # Draw scoring areas with precomputed gradients
    screen.blit(green_gradient, SCORING_RECT.topleft)
    screen.blit(red_gradient, SCORING_RECT.topleft)

    # Optionally mask the beer pint
    if not mask_pint:
        global beer_alpha
        yellow = pygame.Color(int(255 * beer_alpha), int(255 * beer_alpha), 0)
        pygame.draw.circle(screen, yellow, (int(pint_pos[0]), int(pint_pos[1])), pint_radius)
        pygame.draw.circle(screen, WHITE, (int(pint_pos[0]), int(pint_pos[1])), pint_radius + 2, 2)


# Precompute gradient surfaces
def create_gradient_surface(points, start_color, end_color, reference_point):
    """Generate a gradient surface for a triangular region."""
    max_distance = max(math.dist(reference_point, p) for p in points)
    surface = pygame.Surface((SCORING_RECT.width, SCORING_RECT.height), pygame.SRCALPHA)

    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            world_x = SCORING_RECT.left + x
            world_y = SCORING_RECT.top + y
            if point_in_polygon((world_x, world_y), points):
                distance = math.dist((world_x, world_y), reference_point)
                factor = min(distance / max_distance, 1.0)
                color = interpolate_color(start_color, end_color, factor)
                surface.set_at((x, y), color + (255,))  # Add alpha
    return surface

def interpolate_color(start_color, end_color, factor):
    """Interpolate between two colors."""
    return tuple(int(start + (end - start) * factor) for start, end in zip(start_color, end_color))

# PINT_MOVEMENTS
def handle_mouse_input():
    """Handle mouse interactions with the pint."""
    global pint_pos, pint_velocity, launched, waiting_for_mouse
    global ideal_velocity, ideal_pos

    mouse_pos = pygame.mouse.get_pos()
    distance = math.dist(mouse_pos, START_POS)
    if waiting_for_mouse:
        if distance <= pint_radius:  # Mouse touching the pint
            waiting_for_mouse = False
    elif distance <= FREE_ZONE_RADIUS:
        pint_pos[0], pint_pos[1] = mouse_pos
        ideal_pos[0], ideal_pos[1] = mouse_pos
    else:
        pint_velocity = calculate_velocity(pint_pos, mouse_pos)
        ideal_velocity = pint_velocity.copy()
        if perturbation_active:
            apply_perturbation()
        apply_noise()
        launched = True


def calculate_velocity(start_pos, mouse_pos):
    dx = mouse_pos[0] - start_pos[0]
    dy = mouse_pos[1] - start_pos[1]
    speed = math.sqrt(dx**2 + dy**2) / 10
    angle = math.atan2(dy, dx)
    return [speed * math.cos(angle), speed * math.sin(angle)]


def apply_friction():
    global pint_velocity, ideal_velocity, friction
    pint_velocity[0] *= friction
    pint_velocity[1] *= friction

    ideal_velocity[0] *= friction
    ideal_velocity[1] *= friction


def sample_random_noise():
    global noise_mean, noise_std
    return np.random.normal(noise_mean, noise_std)


def update_perturbation():
    """Adjust the perturbation force based on gradual or sudden mode."""
    global perturbation_force, trial_in_block

    if gradual_perturbation and perturbation_active:
        # Increment force every 3 trials (or however you want to adjust the frequency)
        if trial_in_block % 3 == 0 and trial_in_block != 0:
            perturbation_force += force_increment  # Increase perturbation force gradually after each set of 3 trials
            print(f"Gradual perturbation force updated to: {perturbation_force}")
    # Sudden perturbation: No updates needed (force remains constant)


def apply_perturbation():
    """Apply perturbation to the pint's movement."""
    global pint_velocity, perturbation_force
    if perturbation_active:
        pint_velocity[0] += perturbation_force  # Add rightward force


def apply_noise():
    global pint_velocity, noise_instance
    pint_velocity[0] += noise_instance


# CHECK & SCORE
def check_stopped():
    global stopped, launched
    if abs(pint_velocity[0]) < 0.1 and abs(pint_velocity[1]) < 0.1 and launched:
        stopped = True
        launched = False


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[0]
    for i in range(1, n + 1):
        sx, sy = polygon[i % n]
        if y > min(py, sy):
            if y <= max(py, sy):
                if x <= max(px, sx):
                    if py != sy:
                        xinters = (y - py) * (sx - px) / (sy - py) + px
                    if px == sx or x <= xinters:
                        inside = not inside
        px, py = sx, sy
    return inside


def calculate_score():
    """Calculate and update the score."""
    global pint_pos, stopped, end_pos, score, trial_counter, trial_positions, previous_win, previous_pos, \
        previous_trajectory, feedback_type, perturbation_active, gradual_perturbation, perturbation_force

    global ideal_trajectory, ideal_pos

    last_trajectory.append(pint_pos.copy())
    ideal_trajectory.append(ideal_pos.copy())

    if stopped:  # Only calculate score once per trial
        trial_counter += 1
        trial_score = 0

        if point_in_polygon(pint_pos, GREEN_TRIANGLE):
            reference_point = SCORING_RECT.topleft
            distance = math.dist(pint_pos, reference_point)
            max_distance = max(math.dist(p, reference_point) for p in GREEN_TRIANGLE)
            trial_score = calculate_edge_score(distance, max_distance)
        elif point_in_polygon(pint_pos, RED_TRIANGLE):
            reference_point = SCORING_RECT.bottomright
            distance = math.dist(pint_pos, reference_point)
            max_distance = max(math.dist(p, reference_point) for p in RED_TRIANGLE)
            trial_score = -calculate_edge_score(distance, max_distance)
        elif not TABLE_RECT.collidepoint(*pint_pos):
            # Penalty for missing
            trial_score = -50
            if feedback_type != "rl":
                display_message("Too far!")

        score += trial_score

        previous_win = point_in_polygon(pint_pos, GREEN_TRIANGLE)
        previous_pos = pint_pos.copy()
        previous_trajectory = last_trajectory.copy()

        # Append trial position and current block number
        trial_positions.append((pint_pos[0], pint_pos[1], current_block))
        # Save current trial results
        record_results(
            trial_number = trial_counter, score = trial_score, total_score = score,
            feedback_type = feedback_type, end_position = pint_pos,
            perturbation_mode = get_perturbation_mode(perturbation_active, gradual_perturbation),
            perturbation_force = perturbation_force,
            noise_mean = noise_mean,
            noise_std = noise_std,
            noise_instance = noise_instance,
            friction = friction
        )
        reset_pint()
        handle_trial_end()


def get_perturbation_mode(perturbation_active, gradual_perturbation):
    if not perturbation_active: return "None"
    if gradual_perturbation: return "Gradual"
    return "Sudden"


def record_results(
        trial_number, score, total_score, feedback_type,
        end_position, perturbation_mode, perturbation_force,
        noise_mean, noise_std, noise_instance, friction
):
    global results

    print(
        f"Trial {trial_number}: score {score}, total score {total_score}, "
        f"feedback type {feedback_type}, end position {end_position}, "
        f"perturbation mode {perturbation_mode}, perturbation force {perturbation_force}, "
        f"noise mean {noise_mean}, noise std {noise_std}, noise instance {noise_instance}, "
        f"friction {friction}"
    )

    results.append(pd.DataFrame({
        "Trial": [trial_number],
        "Score": [score],
        "TotalScore": [total_score],
        "FeedbackType": [feedback_type],
        "EndPosX": [end_position[0]],
        "EndPosY": [end_position[1]],
        "Perturbation": [perturbation_mode],
        "PerturbationForce": [perturbation_force],
        "NoiseMean": [noise_mean],
        "NoiseStd": [noise_std],
        "NoiseInstance": [noise_instance],
        "Friction": [friction]
    }))


def calculate_edge_score(distance, max_distance):
    """
    Calculate the score based on distance to the reference point.
    100 points for the closest edge, 10 points for the farthest edge.
    """
    normalized_distance = min(distance / max_distance, 1.0)  # Normalize to [0, 1]
    return int(100 - 90 * normalized_distance)  # Scale between 100 and 10


def display_message(text, length = 1000):
    message = font.render(text, True, BLACK)
    screen.blit(message, (SCREEN_WIDTH // 2 - message.get_width() // 2, SCREEN_HEIGHT // 2 - message.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(length)


def reset_pint():
    """Reset the pint to the starting position."""
    global pint_pos, end_pos, last_trajectory, pint_velocity, launched, stopped, waiting_for_mouse, trajectory
    global ideal_velocity, ideal_pos, ideal_trajectory
    pint_pos[:] = START_POS
    pint_velocity[:] = [0, 0]

    ideal_pos[:] = START_POS
    ideal_velocity[:] = [0, 0]
    ideal_trajectory = []

    launched = False
    stopped = False
    waiting_for_mouse = True
    last_trajectory = []


#TASK 1: IMPLEMENT FEEDBACK MODES
def draw_feedback():
    """Display feedback based on the feedback type."""

    global feedback_type

    # Reset
    pygame.draw.circle(screen, BLACK, START_POS, FREE_ZONE_RADIUS, 3)

    if feedback_type is None:
        return

    if feedback_type == "trajectory":
        # Draw the trajectory of the previous throw
        for point in previous_trajectory:
            pygame.draw.circle(screen, WHITE, (int(point[0]), int(point[1])), 2, 2)

    if feedback_type == "rl":
        # Draw a circle around the throw point
        color = GREEN_LAMP if previous_win else RED_LAMP
        pygame.draw.circle(screen, color, START_POS, FREE_ZONE_RADIUS, 10)

    if feedback_type == "endpos":
        # Show the final position of the pint
        if previous_pos is not None:
            pygame.draw.circle(screen, WHITE, (int(previous_pos[0]), int(previous_pos[1])), pint_radius + 2, 2)



# Precompute gradient surfaces
green_gradient = create_gradient_surface(GREEN_TRIANGLE, DARK_GREEN, LIGHT_GREEN, SCORING_RECT.topleft)
red_gradient = create_gradient_surface(RED_TRIANGLE, DARK_RED, LIGHT_RED, SCORING_RECT.bottomright)


#Design Experiment
def setup_block(block_number):
    """Set up block parameters."""
    global perturbation_active, feedback_mode, feedback_type, perturbation_force, trial_in_block, gradual_perturbation
    global friction, friction_decrease
    global noise_active, noise_mean, noise_std, noise_instance
    global beer_alpha, beer_alpha_decrease

    block = block_structure[block_number - 1]
    feedback_type = block['feedback'] if block['feedback'] else None
    feedback_mode = feedback_type is not None

    drink_beer = block.get("drink_beer", False)
    if drink_beer:
        display_message("Drinking beer!", length = 2000)
        beer_alpha -= beer_alpha_decrease
        if friction + friction_decrease < 1:
            friction += friction_decrease

    noise_active = block.get("noise_active", False)
    noise_mean = block.get("noise_mean", 0)
    noise_std = block.get("noise_std", 0)

    # Generate new perturbation noise
    if noise_active:
        noise_instance = sample_random_noise()
    else:
        noise_instance = 0

    perturbation_active = block['perturbation']
    trial_in_block = 0

    # Apply global perturbation mode to set gradual or sudden
    if perturbation_active:
        if block['gradual']:  # Gradual perturbation
            gradual_perturbation = True
            perturbation_force = block.get('initial_force', 0)  # Use the initial force for gradual perturbation
        else:  # Sudden perturbation
            gradual_perturbation = False
            perturbation_force = block.get('sudden_force', 10.0)  # Use the sudden force for sudden perturbation
    else:
        # Reset perturbation force if no perturbation is applied
        perturbation_force = 0


def handle_trial_end():
    """Handle end-of-trial events."""
    global trial_in_block, current_block, running, noise_instance

    trial_in_block += 1

    # Update perturbation force for gradual perturbation
    if perturbation_active and gradual_perturbation:
        update_perturbation()

    # Generate new perturbation noise
    if noise_active:
        noise_instance = sample_random_noise()
    else:
        noise_instance = 0

    # Transition to the next block if trials in the current block are complete
    if trial_in_block >= block_structure[current_block - 1]['num_trials']:
        current_block += 1
        if current_block > len(block_structure):
            running = False  # End experiment
        else:
            setup_block(current_block)


# TASK1: Define the experiment blocks

# 10 trials without perturbation
# 30 trials with gradual perturbation
# 10 trials without perturbation

small_noise_mean = 0
small_noise_std = 1

medium_noise_mean = 1.5
medium_noise_std = 2

large_noise_mean = 2.5
large_noise_std = 3

sudden_force = 2
n_trials_no_perturbation = 10
n_trials_perturbation = 30
feedback_setting = "endpos"

block_structure = [
    # 1
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": False,
        "noise_active": False,
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_perturbation,
        "perturbation": True, "gradual": False, "sudden_force": sudden_force,
        "drink_beer": False,
        "noise_active": False
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": False,
        "noise_active": False
    },

    # 2
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": True,
        "noise_active": True, "noise_mean": small_noise_mean, "noise_std": small_noise_std
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_perturbation,
        "perturbation": True, "gradual": False, "sudden_force": sudden_force,
        "drink_beer": False,
        "noise_active": True, "noise_mean": small_noise_mean, "noise_std": small_noise_std
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": False,
        "noise_active": True, "noise_mean": small_noise_mean, "noise_std": small_noise_std
    },

    # 3
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": True,
        "noise_active": True, "noise_mean": medium_noise_mean, "noise_std": medium_noise_std
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_perturbation,
        "perturbation": True, "gradual": False, "sudden_force": sudden_force,
        "drink_beer": False,
        "noise_active": True, "noise_mean": medium_noise_mean, "noise_std": medium_noise_std
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": False,
        "noise_active": True, "noise_mean": medium_noise_mean, "noise_std": medium_noise_std
    },

    # 4
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": True,
        "noise_active": True, "noise_mean": large_noise_mean, "noise_std": large_noise_std
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_perturbation,
        "perturbation": True, "gradual": False, "sudden_force": sudden_force,
        "drink_beer": False,
        "noise_active": True, "noise_mean": large_noise_mean, "noise_std": large_noise_std
    },
    {
        "feedback": feedback_setting, "num_trials": n_trials_no_perturbation,
        "perturbation": False,
        "drink_beer": False,
        "noise_active": True, "noise_mean": large_noise_mean, "noise_std": large_noise_std
    }
]

current_block = 1
setup_block(current_block)

# Main game loop
clock = pygame.time.Clock()
running = True
while running:
    # Determine if the beer pint should be masked
    mask_pint = launched and feedback_mode and feedback_type in ('trajectory', 'rl', 'endpos')

    # Draw playfield with optional masking
    draw_playfield(mask_pint=mask_pint)

    draw_feedback()

    # Display score (only for feedbacks where score is not relevant)
    if feedback_type not in ('rl', 'endpos', 'trajectory'):
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

    # Handle Keyboard events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_4:
                perturbation_mode = True
            elif event.key == pygame.K_5:
                perturbation_mode = False
            elif event.key == pygame.K_1:
                feedback_type = 'trajectory'
                feedback_mode = True
            elif event.key == pygame.K_2:
                feedback_type = 'endpos'
                feedback_mode = True
            elif event.key == pygame.K_3:
                feedback_type = 'rl'
                feedback_mode = True
            elif event.key == pygame.K_0:
                feedback_type = None
                feedback_mode = False
            elif event.key == pygame.K_i:  # Press 'i' to toggle info display
                show_info = not show_info
            elif event.key == pygame.K_SPACE:  # Start the next experimental block
                current_block += 1
                if current_block > len(block_structure):
                    running = False  # End the experiment
                else:
                    setup_block(current_block)
    if not launched:
        handle_mouse_input()
    else:
        pint_pos[0] += pint_velocity[0]
        pint_pos[1] += pint_velocity[1]

        ideal_pos[0] += ideal_velocity[0]
        ideal_pos[1] += ideal_velocity[1]

        apply_friction()
        check_stopped()
        calculate_score()

    # Draw feedback if applicable
    # draw_feedback()

    if show_info:
        # Idealized trajectory with no noise or perturbations (sanity check)
        for point in ideal_trajectory:
            pygame.draw.circle(screen, WHITE, (int(point[0]), int(point[1])), 2, 2)

        fb_info_text = font.render(f"Feedback: {feedback_type}", True, BLACK)
        pt_info_text = font.render(f"Perturbation:{perturbation_active}", True, BLACK)
        pf_info_text = font.render(f"Perturbation_force:{perturbation_force}", True, BLACK)
        fr_info_text = font.render(f"Friction:{friction}", True, BLACK)
        tib_text = font.render(f"Trial_in_block: {trial_in_block}", True, BLACK)
        noise_text = font.render(f"Noise mean: {noise_mean}, noise std: {noise_std}, noise inst: {noise_instance}", True, BLACK)
        screen.blit(fb_info_text, (10, 60))
        screen.blit(pt_info_text, (10, 90))
        screen.blit(pf_info_text, (10, 120))
        screen.blit(tib_text, (10, 150))
        screen.blit(fr_info_text, (10, 180))
        screen.blit(noise_text, (10, 210))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

#TASK 2: PLOT Hitting patterns for all feedbacks
# Plot results (hitting patterns on table + end score) grouped by feedback type
feedback_blocks = {
    'trajectory': [4, 5, 6],
    'endpos': [7, 8, 9],
    'rl': [10, 11, 12],
    None: [1, 2, 3]  # Normal feedback      
}
#use trial_positions


def get_file_name(prefix: str = "logs"):
    dt = datetime.datetime.now()
    date = str(dt.date())
    time = str(dt.time()).split(".")[0].replace(":", "-")
    return prefix + "_" + date.replace("-", "_") + "_" + time.replace("-", "_")


results = pd.concat(results)
results.reset_index(drop = True, inplace = True)
results.to_csv(f"{get_file_name()}.csv", index = False)
