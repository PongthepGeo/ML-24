#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#-----------------------------------------------------------------------------------------#

# Define the simple quadratic function g(w) = w^2
def g(w):
    return w ** 2

# Define the gradient of the function (which is 2w for g(w) = w^2)
def gradient(w):
    return 2 * w

# Local optimization using a constant step length with a tolerance
def local_optimization(initial_point, step_length, num_iterations, tolerance):
    w = initial_point
    points = [w]  # List to store points for each iteration
    for k in range(num_iterations):
        grad = gradient(w)
        print(f"Iteration {k}: w = {w}, Gradient = {grad}, g(w) = {g(w)}") # Print gradient at each step
        w_new = w - step_length * grad
        # Allow small tolerance for improvement
        if abs(g(w_new) - g(w)) < tolerance:
            break
        w = w_new
        points.append(w)
    return w, points

# Updated Parameters for the simplified quadratic function
initial_point = 10  # Starting point
step_length = 0.1   # Step size
num_iterations = 50  # Maximum number of iterations
tolerance = 1e-6  # Tolerance for stopping

# Run the optimization
final_point, points = local_optimization(initial_point, step_length, num_iterations, tolerance)

# Output the results
print(f"Final point: {final_point}")
print(f"Function value at final point (g(w)): {g(final_point)}")
print(f"Sequence of points: {points}")

U.plot_optimization_path(g, points)