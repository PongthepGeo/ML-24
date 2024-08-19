#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
#-----------------------------------------------------------------------------------------#

# Number of points to sample
num_points = 10
# Generate angles for counterclockwise points (from 0 to 2*pi, excluding 2*pi)
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
# Convert angles to Cartesian coordinates on the unit circle
x_circle = np.cos(angles)
y_circle = np.sin(angles)
# Original weight and bubble tea ranges
weight_min, weight_max = 40, 150
tea_min, tea_max = 0, 10
# Scale circle points to the original weight and bubble tea ranges
weights = ((x_circle + 1) / 2) * (weight_max - weight_min) + weight_min
bubble_teas = ((y_circle + 1) / 2) * (tea_max - tea_min) + tea_min
normalized_weights = U.normalize(weights, weight_min, weight_max, -5, 5)
normalized_bubble_teas = U.normalize(bubble_teas, tea_min, tea_max, -5, 5)
# Create the meshgrid (already in [-5, 5] range)
x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
# Calculate u and v based on the equations
u = -y / np.sqrt(x**2 + y**2)
v = x / np.sqrt(x**2 + y**2)

#-----------------------------------------------------------------------------------------#

U.vector(normalized_weights, normalized_bubble_teas, x, y, u, v, weights, bubble_teas)

#-----------------------------------------------------------------------------------------#