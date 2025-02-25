import pygame
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime


# Game parameters
SCREEN_X, SCREEN_Y = 1920, 1080  # your screen resolution
WIDTH, HEIGHT = (
    SCREEN_X,
    SCREEN_Y,
)  # be aware of monitor scaling on windows (150%)
CIRCLE_SIZE = 20
TARGET_SIZE = CIRCLE_SIZE
TARGET_RADIUS = 300
MASK_RADIUS = 0.66 * TARGET_RADIUS
START_POSITION = (WIDTH // 2, HEIGHT // 2)
START_ANGLE = 0

# List the target starting angles here
TARGET_ANGLES = [-15, -75, (-75 - 15) / 2, -120]

PERTURBATION_ANGLE = 30
TIME_LIMIT = 1000  # time limit in ms

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Reaching Game")


# Initialize game metrics
score = 0
attempts = 0
new_target = None
start_time = 0

# Setup for blocks
n_perturbation = 60
n_no_perturbation = 20
block_len = n_no_perturbation + n_perturbation + n_no_perturbation

block1_start = 0
block2_start = block_len
block3_start = block_len * 2
block4_start = block_len * 3
block5_start = block_len * 4

ATTEMPTS_LIMIT = block_len * 5

new_target = None
start_target = math.radians(START_ANGLE)
move_faster = False
clock = pygame.time.Clock()

# Initialize game modes
mask_mode = True
target_mode = "fix"  # Mode for angular shift of target: random, fix, dynamic
perturbation_mode = False
perturbation_type = (
    "sudden"  # Mode for angular shift of controll: random, gradual or sudden
)
perturbation_angle = math.radians(
    PERTURBATION_ANGLE
)  # Angle between mouse_pos and circle_pos
perturbed_mouse_angle = 0
gradual_step = 0
gradual_attempts = 1
perturbation_rand = random.uniform(-math.pi / 4, +math.pi / 4)

error_angle_logs = np.zeros(ATTEMPTS_LIMIT)  # List to store error angles
move_faster_logs = np.zeros(ATTEMPTS_LIMIT)
time_logs = np.zeros(ATTEMPTS_LIMIT)
mouse_pos_logs = []
circle_pos_logs = []
attempt_logs = []
# Flag for showing mouse position and deltas
show_mouse_info = False


# Function to generate a new target position
def generate_target_position():
    if target_mode == "random":
        angle = random.uniform(0, 2 * math.pi)

    elif target_mode == "fix":
        angle = start_target

    new_target_x = WIDTH // 2 + TARGET_RADIUS * math.sin(angle)
    new_target_y = HEIGHT // 2 + TARGET_RADIUS * -math.cos(
        angle
    )  # zero-angle at the top
    return [new_target_x, new_target_y]


# Function to check if the current target is reached
def check_target_reached():
    if new_target:
        distance = math.hypot(
            circle_pos[0] - new_target[0], circle_pos[1] - new_target[1]
        )
        return distance <= CIRCLE_SIZE
    return False


# Function to check if player is at starting position and generate new target
def at_start_position_and_generate_target(mouse_pos):
    distance = math.hypot(
        mouse_pos[0] - START_POSITION[0], mouse_pos[1] - START_POSITION[1]
    )
    if distance <= CIRCLE_SIZE:
        return True
    return False


def get_delta_angle(reference: np.array, other: np.array) -> float:
    """get delta angle between two 2D positions

    Args:
        reference (np.array): (x, y)
        other (np.array): (x, y)

    Returns:
        float: _description_
    """
    reference_angle = np.arctan2(*reference[::-1])
    other_angle = np.arctan2(*other[::-1])
    angle = other_angle - reference_angle
    return angle


def get_file_name(prefix: str = "logs"):
    dt = datetime.datetime.now()
    date = str(dt.date())
    time = str(dt.time()).split(".")[0].replace(":", "-")
    return prefix + "_" + date.replace("-", "_") + "_" + time.replace("-", "_")


# Main game loop
running = True
while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Press 'esc' to close the experiment
                running = False
            elif event.key == pygame.K_4:  # Press '4' to test pertubation_mode
                perturbation_mode = True
            elif event.key == pygame.K_5:  # Press '5' to end pertubation_mode
                perturbation_mode = False
            elif event.key == pygame.K_h:  # Press 'h' to toggle mouse info display
                show_mouse_info = not show_mouse_info

    # Design experiment

    # A single block:
    #   - 20 attempts without perturbation
    #   - 60 attempts with perturbation
    #   - 20 attempts without perturbation

    # Block 1
    if attempts == block1_start:
        START_ANGLE = TARGET_ANGLES[0]
        start_target = math.radians(START_ANGLE)
        perturbation_mode = False
    elif attempts == block1_start + n_no_perturbation:
        perturbation_mode = True
        perturbation_type = "sudden"
    elif attempts == block1_start + n_no_perturbation + n_perturbation:
        perturbation_mode = False

    # Block 2
    elif attempts == block2_start:
        START_ANGLE = TARGET_ANGLES[1]
        start_target = math.radians(START_ANGLE)
        perturbation_mode = False
    elif attempts == block2_start + n_no_perturbation:
        perturbation_mode = True
        perturbation_type = "sudden"
    elif attempts == block2_start + n_no_perturbation + n_perturbation:
        perturbation_mode = False

    # Block 3
    elif attempts == block3_start:
        START_ANGLE = TARGET_ANGLES[2]
        start_target = math.radians(START_ANGLE)
        perturbation_mode = False
    elif attempts == block3_start + n_no_perturbation:
        perturbation_mode = True
        perturbation_type = "sudden"
    elif attempts == block3_start + n_no_perturbation + n_perturbation:
        perturbation_mode = False

    # Block 4
    elif attempts == block4_start:
        START_ANGLE = TARGET_ANGLES[3]
        start_target = math.radians(START_ANGLE)
        perturbation_mode = False
    elif attempts == block4_start + n_no_perturbation:
        perturbation_mode = True
        perturbation_type = "sudden"
    elif attempts == block4_start + n_no_perturbation + n_perturbation:
        perturbation_mode = False

    # Block 5: repeat block 4 with perturbation angle in the opposite direction
    if attempts == block5_start:
        START_ANGLE = TARGET_ANGLES[3]
        start_target = math.radians(START_ANGLE)
        perturbation_mode = False
    elif attempts == block5_start + n_no_perturbation:
        # Reverse perturbation angle
        perturbation_angle = -perturbation_angle
        perturbation_mode = True
        perturbation_type = "sudden"
    elif attempts == block5_start + n_no_perturbation + n_perturbation:
        perturbation_mode = False

    # End
    elif attempts >= ATTEMPTS_LIMIT:
        running = False

    # Hide the mouse cursor
    pygame.mouse.set_visible(False)
    # Get mouse position
    mouse_pos = pygame.mouse.get_pos()

    # Calculate distance from START_POSITION to mouse_pos
    deltax = mouse_pos[0] - START_POSITION[0]
    deltay = mouse_pos[1] - START_POSITION[1]
    distance = math.hypot(deltax, deltay)
    mouse_angle = math.atan2(deltay, deltax)

    # TASK1: CALCULATE perturbed_mouse_pos
    # PRESS 'h' in game for a hint
    if perturbation_mode:
        if perturbation_type == "sudden":
            perturbed_mouse_angle = perturbation_angle

        rot_mat = np.array([
            [np.cos(perturbed_mouse_angle), -np.sin(perturbed_mouse_angle)],
            [np.sin(perturbed_mouse_angle), np.cos(perturbed_mouse_angle)],
        ])
        perturbed_mouse_pos = (
            rot_mat @ (np.array(mouse_pos) - START_POSITION) + START_POSITION
        )
        circle_pos = perturbed_mouse_pos.tolist()
    else:
        circle_pos = pygame.mouse.get_pos()

    # log dynamics
    attempt_logs.append(attempts)
    mouse_pos_logs.append(mouse_pos)
    circle_pos_logs.append(circle_pos)

    # Check if target is hit or missed
    # hit if circle touches target's center
    if check_target_reached():
        score += 1
        attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a hit
        new_target = np.array(new_target) - np.array(START_POSITION)
        circle_pos = np.array(circle_pos) - np.array(START_POSITION)
        error_angle = np.rad2deg(get_delta_angle(new_target, circle_pos))
        error_angle_logs[attempts - 1] = error_angle
        current_time = pygame.time.get_ticks()
        time_logs[attempts - 1] = current_time - start_time

        new_target = None  # Set target to None to indicate hit
        start_time = 0  # Reset start_time after hitting the target
        if perturbation_type == "gradual" and perturbation_mode:
            gradual_attempts += 1

    # miss if player leaves the target_radius + 1% tolerance
    elif (
        new_target
        and math.hypot(
            circle_pos[0] - START_POSITION[0], circle_pos[1] - START_POSITION[1]
        )
        > TARGET_RADIUS * 1.01
    ):
        attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a miss
        new_target = np.array(new_target) - np.array(START_POSITION)
        circle_pos = np.array(circle_pos) - np.array(START_POSITION)
        error_angle = np.rad2deg(get_delta_angle(new_target, circle_pos))
        error_angle_logs[attempts - 1] = error_angle
        current_time = pygame.time.get_ticks()
        time_logs[attempts - 1] = current_time - start_time

        new_target = None  # Set target to None to indicate miss
        start_time = 0  # Reset start_time after missing the target

        if perturbation_type == "gradual" and perturbation_mode:
            gradual_attempts += 1

    # Check if player moved to the center and generate new target
    if not new_target and at_start_position_and_generate_target(mouse_pos):
        new_target = generate_target_position()
        move_faster = False
        start_time = pygame.time.get_ticks()  # Start the timer for the attempt

    # Check if time limit for the attempt is reached
    current_time = pygame.time.get_ticks()
    if start_time != 0 and (current_time - start_time) > TIME_LIMIT:
        move_faster = True
        start_time = 0  # Reset start_time
        move_faster_logs[attempts] = 1

    # Show 'MOVE FASTER!'
    if move_faster:
        font = pygame.font.Font(None, 36)
        text = font.render("MOVE FASTER!", True, RED)
        text_rect = text.get_rect(center=(START_POSITION))
        screen.blit(text, text_rect)

    # Generate playing field
    # Draw current target
    if new_target:
        pygame.draw.circle(screen, BLUE, new_target, TARGET_SIZE // 2)

    # Draw circle cursor
    if mask_mode:
        if distance < MASK_RADIUS:
            pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)
    else:
        pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)

    # Draw start position
    pygame.draw.circle(screen, WHITE, START_POSITION, 5)

    # Show score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Show attempts
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Attempts: {attempts} / {ATTEMPTS_LIMIT}", True, WHITE)
    screen.blit(score_text, (10, 30))

    if show_mouse_info:
        mouse_info_text = font.render(
            f"Mouse: x={mouse_pos[0]}, y={mouse_pos[1]}", True, WHITE
        )
        delta_info_text = font.render(f"Delta: Δx={deltax}, Δy={deltay}", True, WHITE)
        mouse_angle_text = font.render(
            f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, WHITE
        )
        pertubation_mode_text = font.render(
            f"Pertubation_mode: {perturbation_mode}, {perturbation_type}", True, WHITE
        )
        target_position_text = font.render(
            f"Target position: {START_ANGLE}", True, WHITE
        )
        screen.blit(mouse_info_text, (10, 60))
        screen.blit(delta_info_text, (10, 90))
        screen.blit(mouse_angle_text, (10, 120))
        screen.blit(pertubation_mode_text, (10, 150))
        screen.blit(target_position_text, (10, 180))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()


## TASK 2, CALCULATE, PLOT AND SAVE (e.g. export as .csv) ERRORS from error_angles
np.save(
    get_file_name(),
    {
        "move_faster_logs": move_faster_logs,
        "error_angle_logs": error_angle_logs,
        "time_logs": time_logs,
        "attempt_logs": np.array(attempt_logs),
        "mouse_pos_logs": np.array(mouse_pos_logs) - np.array(START_POSITION),
        "circle_pos_logs": np.array(circle_pos_logs) - np.array(START_POSITION),
    },
)
sys.exit()
