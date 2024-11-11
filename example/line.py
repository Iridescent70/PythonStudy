import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Construct data points
set_distances = [0.8, 1.6, 2.4] * 5  # Set distances for the car to move, repeated 5 times for multiple experiments
odometry_readings = [0.79, 1.58, 2.39] * 5  # Corresponding odometry readings

# Add some random noise to simulate real-world uncertainty
np.random.seed(0)  # Set random seed for reproducibility
noise = np.random.normal(0, 0.05, len(odometry_readings))  # Assume a standard deviation of 0.05 for the noise
odometry_readings = np.array(odometry_readings) + noise

# Fit the data using linear regression
slope, intercept, r_value, p_value, std_err = linregress(set_distances, odometry_readings)

# Print regression results
print(f"Regression equation: y = {slope:.2f}x + {intercept:.2f}")
print(f"Correlation coefficient: {r_value:.2f}")

# Plot data points and regression line
plt.figure(figsize=(10, 6))
plt.scatter(set_distances, odometry_readings, color='blue', label='Experimental Data')
plt.plot(set_distances, slope * np.array(set_distances) + intercept, color='red', label=f'Regression Line (y={slope:.2f}x+{intercept:.2f})')
plt.xlabel('Set Distance (m)')
plt.ylabel('Odometry Reading (m)')
plt.title('Relationship Between Car Movement Distance and Odometry Reading')
plt.legend()
plt.grid(True)
plt.show()

# Validate the model
test_distances = [0.8, 1.6, 2.4]
predicted_readings = slope * np.array(test_distances) + intercept
actual_readings = [0.79, 1.58, 2.39]  # Assume these are the actual readings from previous experiments for validation

# Output the comparison between predicted and actual values
for test_distance, predicted_reading, actual_reading in zip(test_distances, predicted_readings, actual_readings):
    print(f"Set Distance: {test_distance:.1f}m, Predicted Reading: {predicted_reading:.2f}m, Actual Reading: {actual_reading:.2f}m")