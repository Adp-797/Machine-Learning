import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize

dataset=pd.read_csv('Enter the path to your selected dataset')
print(dataset.shape)
print(dataset.info())
print(dataset.head())
print(dataset.describe())

#Visualizing the dataset
#Box plot
sns.boxplot(data=dataset)

#Dataset with only numerical attributes
#Since K-Means is performed only on numerical values, the attributes Age, Credit amount, and Duration have been selected
cluster_data=dataset[['Age','Credit amount','Duration']]
print(cluster_data)
plt.scatter(cluster_data['Age'],cluster_data['Credit amount'],color='blue')
plt.scatter(cluster_data['Age'],cluster_data['Duration'],color='green')

#Normalization of the dataset
cluster_data['Age']=cluster_data['Age']/100;
cluster_data['Credit amount']=cluster_data['Credit amount']/100000;
cluster_data['Duration']=cluster_data['Duration']/100;
cluster_data.head()

#Finding the optimum number of clusters using the Elbow method
clusters_r= range(1,10)
distortion=[]
for c in clusters_r:
  km= KMeans(n_clusters=c) #no of clusters to form and no of centroids to be generated
  km.fit(cluster_data)
  distortion.append(km.inertia_) #sum of squared distances of the points to the closest centroid
print(distortion)  
print(km)

plt.figure()
plt.plot(clusters_r, distortion,marker='o')
plt.xlabel('Cluster range')
plt.ylabel('Distortion')
plt.title('Elbow method')
plt.show()

#Fitting k means for k clusters
km= KMeans(n_clusters=3) #From the above figure, k=3
km.fit(cluster_data)
y_pred= km.fit_predict(cluster_data)
y_pred

#Mapping predicted values into different clusters
cluster_data['cluster']= y_pred
cluster_data.head()

#Grouping into three clusters
cluster_data_0 = cluster_data[cluster_data.cluster==0]
cluster_data_1 = cluster_data[cluster_data.cluster==1]
cluster_data_2 = cluster_data[cluster_data.cluster==2]

#Scatter plots for visualization
#Credit amount vs Age
plt.scatter(cluster_data_0.Age,cluster_data_0['Credit amount'])
plt.scatter(cluster_data_1.Age,cluster_data_1['Credit amount'])
plt.scatter(cluster_data_2.Age,cluster_data_2['Credit amount'])
plt.xlabel('Age')
plt.ylabel('Credit amount')

#Duration vs Credit amount
plt.scatter(cluster_data_0['Credit amount'],cluster_data_0.Duration)
plt.scatter(cluster_data_1['Credit amount'],cluster_data_1.Duration)
plt.scatter(cluster_data_2['Credit amount'],cluster_data_2.Duration)
plt.xlabel('Credit amount')
plt.ylabel('Duration')

#Duration vs Age
plt.scatter(cluster_data_0['Age'],cluster_data_0.Duration)
plt.scatter(cluster_data_1['Age'],cluster_data_1.Duration)
plt.scatter(cluster_data_2['Age'],cluster_data_2.Duration)
plt.xlabel('Age')
plt.ylabel('Duration')

#Heatmap for interpretation
cluster_data['Age']=cluster_data['Age']*100;
cluster_data['Credit amount']=cluster_data['Credit amount']*100000;
cluster_data['Duration']=cluster_data['Duration']*100;

grouped_km=cluster_data.groupby(['cluster']).mean().round(1)
grouped_km
