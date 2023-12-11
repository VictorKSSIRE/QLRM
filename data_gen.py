import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
This File Generates all the data we work with in our linear models
"""


# Parameters for data generation
# Defined in k for price and square footage
num_train = 16
num_test = 4
square_foot_range = (1, 5)
num_bedrooms_range = (1, 5)
num_bathrooms_range = (1, 3)
rent_mean = 3
rent_std = 0.7
# theta_0 = 500 , theta_2 = 500 , theta_3 = 200 prior to data rescaling
theta_0 = 0.5 #Base price
theta_1 = 1 #Price per thousand square foot
theta_2 = 0.5 #Price per bedroom
theta_3 = 0.2 #Price per bathroom

# Data Generation
def generate_data(num_points, seed=None):
    """
    Helper Function that generates data given some linear relationship
    """
    if seed is not None:
        np.random.seed(seed)
    x_1 = np.random.normal(rent_mean, rent_std, num_points)
    x_1 = np.clip(x_1, square_foot_range[0], square_foot_range[1])
    # Define num_bedrooms as a function of living_areas
    # Larger houses can have more bedrooms and bathrooms
    x_2 = np.random.randint(1, np.minimum(num_bedrooms_range[1], (x_1 / 0.5).astype(int)) + 1, num_points)
    x_3 = np.random.randint(1, np.minimum(num_bathrooms_range[1], (x_1 / 0.5).astype(int)) + 1, num_points)
    # Label Calculation
    labels = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_3 + np.random.normal(0, 0.1, num_points)
    return np.column_stack((x_1, x_2, x_3)), labels


# Training data Generation
X_train, y_train = generate_data(num_train, seed=0)
# Testing data Generation
X_test, y_test = generate_data(num_test)


# Create a DF for data
df_train = pd.DataFrame(X_train, columns=['x_1', 'x_2', 'x_3'])
df_train['y'] = y_train
df_test = pd.DataFrame(X_test, columns=['x_1', 'x_2', 'x_3'])
df_test['y'] = y_test

# Save to CSV file
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

# Plotting the data
plt.figure(figsize=(10, 6))

# Scatter plot of data points (X vs. y) in blue
plt.scatter(X_train[:, 0], y_train, color='blue', label='Training')
plt.scatter(X_test[:, 0], y_test, color='red', label='Test')


plt.xlabel('Living Area (sqft)')
plt.ylabel('Price')
plt.title('Data Plot')
plt.legend()
plt.grid(True)
plt.savefig("plot.jpeg")
plt.clf()


