#Understanding the dataset
# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib
# Importing the dataset
dataset = pd.read_csv('/content/iris data.csv') 
print(dataset)
print(dataset.shape)
x=dataset.iloc[:,0:4]
y=dataset['iris_class']
print(x)
print(y)

#Train test split for dummy classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

#Using dummy classifiers
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
clf_dummy = DummyClassifier(random_state=42) 

clf_dummy.fit(x_train, y_train)
y_pred = clf_dummy.predict(x_test) #uniform parameter
print('Accuracy of dummy classifier on test set: {:.2f}'.format(clf_dummy.score(x_test, y_test)*100)+ '%')

confusion_matrix=confusion_matrix(y_test, y_pred)
print(confusion_matrix)

report = classification_report(y_test, y_pred)
print(report)

#Train test split for SVM
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

#Applying 4 kernal functions
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")

#SVC model
for i in range(4):
    # Separate data into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=10)# Train a SVC model using different kernal
    svclassifier = getClassifier(i) 
    svclassifier.fit(x_train, y_train)# Make prediction
    y_pred = svclassifier.predict(x_test)# Evaluate our model
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(svclassifier.score(x_test, y_test)*100)+ '%')
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test,y_pred))
    
#GridsearchCV for Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(x_train,y_train)

print(grid.best_estimator_) #finding the optimal parameters
print(grid.best_params_)

grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
