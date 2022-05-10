import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Enter path of iris dataset') 

x1=dataset.iloc[:,0:2] #sepal length & width as features
y1=dataset['iris_class']
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(x1_train, y1_train)
y1_pred = logreg.predict(x1_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x1_test, y1_test)*100))

confusion_matrix = confusion_matrix(y1_test, y1_pred)
report = classification_report(y1_test, y1_pred)
print(report)

ax = sns.heatmap(confusion_matrix, annot=True, cmap='Greens')

ax.set_title('Confusion matrix with \n Sepal length & width as inputs\n\n');
ax.set_xlabel('\nPredicted Flower Category')
ax.set_ylabel('Actual Flower Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.tick_top()
ax.xaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
ax.yaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])

## Display the visualization of the Confusion Matrix.
plt.show()
