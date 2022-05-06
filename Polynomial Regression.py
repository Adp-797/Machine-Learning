# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Importing the dataset
dataset = pd.read_csv('/content/sample_data/california_housing_test.csv') 
print(dataset)
x=dataset['median_income']
y=dataset['median_house_value']
plt.scatter(x, y, color='red', alpha=0.5)
plt.show()

#reshape to 2D arrays
x_np = np.reshape(x.to_numpy(), (-1,1))
y_np = np.reshape(y.to_numpy(), (-1,1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size = 1/3, random_state = 0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Train using train split
regressor_2 = LinearRegression()
regressor_2.fit(x_train, y_train)

y_pred = regressor_2.predict(x_test)

# Print the coeffiecents (theta_1, theta_2, ...)
print('Coeffecients: ', regressor_2.coef_)

# Print the y-intercept (theta_0)
print('Intercept: ', regressor_2.intercept_)

print('MSE', mean_squared_error(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))

plt.scatter(x_test, y_test, color='red', alpha=0.5)
plt.plot(x_test, y_pred, color='blue',  alpha=0.3)

#Generating Polynomial features
from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(degree=2)
#x_poly = polynomial_features.fit_transform(X)
x_train_poly =  polynomial_features.fit_transform(x_train)
x_test_poly = polynomial_features.transform(x_test)
print(x_train_poly)

plt.scatter(x_train_poly[:,1], x_train_poly[:,2], color='red', alpha=0.5)

#Fitting a linear regressor
# Fit using only the 2nd degree feature
x_train_poly_2 = np.reshape(x_train_poly[:,2], (-1,1))
x_test_poly_2 = np.reshape(x_test_poly[:,2], (-1,1)) 

model_poly = LinearRegression()
model_poly.fit(x_train_poly_2, y_train)

y_poly_pred = model_poly.predict(x_test_poly_2)

# Visualising the Y and X results
plt.scatter(x_test, y_test, color='red', alpha=0.5)
plt.scatter(x_test, y_poly_pred, color='black', alpha=0.5) 
plt.show()

print('MSE', mean_squared_error(y_test, y_poly_pred))
print('MAE', mean_absolute_error(y_test, y_poly_pred))
