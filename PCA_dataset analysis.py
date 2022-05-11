#Reading the dataset student.mat
#Make sure that the attributes are numerical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('/content/student-mat.csv',sep=';')
print(data)

#Taking only numerical attributes
x=data[['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']]
print(x)
print(x.shape)

#Choosing G3 as target
y=data['G3']
print(y)
print(y.shape)

#Applying linear regression(multiple features) without PCA
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor_2 = LinearRegression()
regressor_2.fit(x_train, y_train)
y_pred=regressor_2.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('MSE',mean_squared_error(y,regressor_2.predict(x)))
print('MSA',mean_absolute_error(y,regressor_2.predict(x)))
print('Accuracy',regressor_2.score(x_test,y_test)*100)

#Finding optimum of PCA components to achieve 95% varation
from sklearn.decomposition import PCA
pca=PCA(0.95)
pca.fit(x)
reduced=pca.transform(x)
print(reduced.shape)
#which means, 6 components are required!

#Applying PCA
from sklearn.decomposition import PCA

# Choosing 6 PCA components

pca = PCA(n_components=6)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1','PC2','PC3','PC4','PC5','PC6'])

print(principalDf)

#Variance
var = pca.explained_variance_ratio_
print(var)

plt.bar(['PC1', 'PC2','PC3','PC4','PC5','PC6'], var)
plt.title('Variance vs PC1, PC2, PC3, PC4, PC5, PC6')
plt.xlabel('Principal Components')
plt.ylabel('Variance')

#Applying Linear Regression using PCA components
from sklearn.model_selection import train_test_split
x2=principalDf
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor_3 = LinearRegression()
regressor_3.fit(x2_train, y2_train)
y_pred2=regressor_3.predict(x2_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('MSE',mean_squared_error(y,regressor_3.predict(x2)))
print('MSA',mean_absolute_error(y,regressor_3.predict(x2)))
print('Accuracy',regressor_3.score(x2_test,y2_test)*100)
