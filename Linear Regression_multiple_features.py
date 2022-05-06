#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Enter the path to your selected dataset!")
print(dataset)
x1=dataset["longitude"] #input
x2=dataset["latitude"] #input
y=dataset["median_house_value"] #output
print(x1)
print(x2)
print(y)

#Shape & reshaping functions
print(x1.shape)
print(x2.shape)
print(y.shape)
x1=np.reshape(x1.to_numpy(),(-1,1))
x2=np.reshape(x2.to_numpy(),(-1,1))
y=np.reshape(y.to_numpy(),(-1,1))
print(x1.shape)
print(x2.shape)
print(y.shape)

#3DScatterplot
ax = plt.axes(projection ="3d")
ax.scatter3D(x1, x2, y, color = "red")
plt.title("Median house value vs Longitude & Latitude")
ax.set_xlabel("Longitude->")
ax.set_ylabel("Latitude->")
ax.set_zlabel("Median house value")
plt.show()

#Storing two inputs
x=dataset.iloc[:,0:2] #extracting longitude & latitude
y=dataset['median_house_value']
print(x)
print(y)
print(x.shape)
print(y.shape)

#linear regression
from sklearn.linear_model import LinearRegression
regressor_1= LinearRegression()
regressor_1.fit(x,y) 

#Visualizing the trained model
ax = plt.axes(projection ="3d")
ax.scatter3D(x.iloc[:,0], x.iloc[:,1], y, color = "blue")#3D plot of 2i/p & 1o/p

X1, X2 = np.meshgrid(x.iloc[:,0], x.iloc[:,1])#meshgrid gives a rectangular grid of x
y_pred = regressor_1.coef_[0] * X1 + regressor_1.coef_[1] * X2 + regressor_1.intercept_ #plane equation aX1+bX2+c=y, where c is the intercept, a&b are the coefficients of x1 & x2
ax.plot_surface(X1, X2, y_pred, color='brown')

plt.title("Median house value vs Longitude & Latitude")
ax.set_xlabel("Longitude->")
ax.set_ylabel("Latitude->")
ax.set_zlabel("Median house value")
plt.show()

#Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Visualizing train and test data
ax = plt.axes(projection ="3d")
ax.scatter3D(x_train.iloc[:,0], x_train.iloc[:,1], y_train, color='green')
ax.scatter3D(x_test.iloc[:,0], x_test.iloc[:,1], y_test, color='blue')
plt.title("Median house value vs Longitude & Latitude")
ax.set_xlabel("Longitude->")
ax.set_ylabel("Latitude->")
ax.set_zlabel("Median house value")
plt.show()

#train using train split
regressor_2 = LinearRegression()
regressor_2.fit(x_train, y_train)
y_pred=regressor_2.predict(x_test)

#Visualizing y_test &y_pred
ax = plt.axes(projection ="3d")

ax.scatter3D(x_test.iloc[:,0], x_test.iloc[:,1], y_test, color='green')
ax.scatter3D(x_test.iloc[:,0], x_test.iloc[:,1], y_pred, color='blue')

X1, X2 = np.meshgrid(x_test.iloc[:,0], x_test.iloc[:,1])#meshgrid gives a rectangular grid of x
y_pred = regressor_2.coef_[0] * X1 + regressor_2.coef_[1] * X2 + regressor_2.intercept_ #plane equation aX1+bX2+c=y, where c is the intercept, a&b are the coefficients of x1 & x2
ax.plot_surface(X1, X2, y_pred, color='brown')

plt.title("Median house value vs Longitude & Latitude")
ax.set_xlabel("Longitude->")
ax.set_ylabel("Latitude->")
ax.set_zlabel("Median house value")

plt.title("Median house value vs Longitude & latitude")
plt.show()

#Evaluating performance using MSE & MAE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('MSE',mean_squared_error(y,regressor_2.predict(x)))
print('MSA',mean_absolute_error(y,regressor_2.predict(x)))
print('Accuracy',regressor_2.score(x_test,y_test)*100)
