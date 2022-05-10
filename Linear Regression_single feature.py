#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Enter the path of california dataset")

print(dataset)
x8=dataset['median_income']
y=dataset['median_house_value']
print(x8)
print(y)#median_house_value
print(x8.shape)
print(y.shape)
x8=np.reshape(x8.to_numpy(),(-1,1))
y=np.reshape(y.to_numpy(),(-1,1))
print(x8.shape)
print(y.shape)

plt.scatter(x8,y,color='green')
plt.title("Median house value vs Median Income")
plt.xlabel("Median Income->")
plt.ylabel("Median house value->")
plt.show()

from sklearn.linear_model import LinearRegression

regressor_1= LinearRegression()
regressor_1.fit(x8,y)

plt.scatter(x8,y,color='green')
plt.plot(x8,regressor_1.predict(x8),color='blue')
plt.title("Median house value vs Median Income")
plt.xlabel("Median Income->")
plt.ylabel("Median house value->")
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x8, y, test_size = 1/3, random_state = 0)

plt.scatter(x_train, y_train, color='green')
plt.scatter(x_test, y_test, color='blue')
plt.title("Median house value vs Median Income")
plt.xlabel("Median Income->")
plt.ylabel("Median house value->")
plt.show()

regressor_2 = LinearRegression()
regressor_2.fit(x_train, y_train)
y_pred=regressor_2.predict(x_test)

plt.scatter(x_test, y_test, color='green')
plt.scatter(x_test, y_pred, color='blue')
plt.plot(x_test, y_pred, color='black')
plt.title("Median house value vs Median Income")
plt.xlabel("Median Income->")
plt.ylabel("Median house value->")
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('MSE',mean_squared_error(y,regressor_1.predict(x8)))
print('MAE',mean_absolute_error(y,regressor_1.predict(x8)))
print('Accuracy',regressor_2.score(x_test,y_test)*100)







