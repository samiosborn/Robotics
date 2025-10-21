# keyboard_control.py
import config
from kinematics.forward import forward_kinematics
from visualisation.plot2d_interactive import plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

# Track joint angles
current_joint_angles = np.array(config.JOINT_OFFSETS, dtype=float)

# Button press to angle change
angle_delta = np.deg2rad(10)

# Interactive mode on
plt.ion()

# Initialise figure and axes with subplots
fig, ax = plt.subplots()

# Instructions
print("Press keys: q/a, w/s, e/d to move joints.")

# Exit
print("Press x to exit")

# Initialise Pygame
pygame.init()

# Set mode display size
display = pygame.display.set_mode((400, 600))

# Set display caption
pygame.display.set_caption("Keyboard Control of 3DOF 2D Robot Arm")

running = True
# Loop 
while running:
    # Pump events from keyboard
    pygame.event.pump()

    # Get all keys currently held
    keys = pygame.key.get_pressed()

    # Link 1 reading
    if keys[pygame.K_q]:
        link_1_multiplier = 1
    elif keys[pygame.K_a]:
        link_1_multiplier = -1
    else: 
        link_1_multiplier = 0

    # Link 2 reading
    if keys[pygame.K_w]:
        link_2_multiplier = 1
    elif keys[pygame.K_s]:
        link_2_multiplier = -1
    else: 
        link_2_multiplier = 0

    # Link 3 reading
    if keys[pygame.K_e]:
        link_3_multiplier = 1
    elif keys[pygame.K_d]:
        link_3_multiplier = -1
    else: 
        link_3_multiplier = 0

    # Joint angles delta
    joint_angles_delta = angle_delta * np.array([link_1_multiplier, 
                                               link_2_multiplier, 
                                               link_3_multiplier])

    # Update link end positions
    positions = forward_kinematics(
        joint_angles = angle_delta * np.array([link_1_multiplier, 
                                               link_2_multiplier, 
                                               link_3_multiplier]), 
        joint_offsets = current_joint_angles,
        link_lengths = config.LINK_LENGTHS, 
        base_position = config.BASE_POSITION
    )

    # Update joint angles
    current_joint_angles += joint_angles_delta

    # Plot robot arm
    plot_2d(positions, ax)
    
    # Quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif keys[pygame.K_x]: 
            running = False

    # Sleep for CPU break
    time.sleep(0.1)

plt.ioff()
plt.close('all')
pygame.quit()
