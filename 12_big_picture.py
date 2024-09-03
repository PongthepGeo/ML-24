import numpy as np
import matplotlib.pyplot as plt

# Define the function to minimize
def g(w):
    return w**2

# Define the gradient of the function
def gradient(w):
    return 2 * w

# Local optimization using a constant step length
def local_optimization(initial_point, step_length, num_iterations):
    w = initial_point
    points = [w]  # List to store points for each iteration
    
    for k in range(num_iterations):
        grad = gradient(w)
        w_new = w - step_length * grad
        
        # Stop if there's no further improvement
        if g(w_new) >= g(w):
            break
        
        w = w_new
        points.append(w)
    
    return w, points

# Parameters
initial_point = 10  # Starting point w_0
step_length = 0.1   # Constant step length alpha
num_iterations = 50  # Maximum number of iterations

# Run the optimization
final_point, points = local_optimization(initial_point, step_length, num_iterations)

# Output the results
print(f"Final point: {final_point}")
print(f"Function value at final point: {g(final_point)}")
print(f"Sequence of points: {points}")

# Plotting
w_values = np.linspace(-12, 12, 400)
g_values = g(w_values)

plt.plot(w_values, g_values, label='g(w) = w^2')
plt.scatter(points, [g(w) for w in points], color='red', label='Optimization Path')
plt.xlabel('w')
plt.ylabel('g(w)')
plt.title('Local Optimization Path with Constant Step Length')
plt.legend()
plt.show()