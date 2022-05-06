from matplotlib.image import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# upload and read your image here
image =  imread('Enter the path to your selected image')

# image shape
print(image.shape)

# plot the image
plt.imshow(image)

X = image.reshape(-1, 3) #this is done to convert 3D array to 2D with RGB colors

# kmeans clustering
kmeans = KMeans(n_clusters=4, random_state=101).fit(X)

# cluster centers
print(kmeans.cluster_centers_)

# labels of each of row of X
print(kmeans.labels_)

# shape of the kmeans.labels_ array
print(kmeans.labels_.shape)

seg_img = []

for label in kmeans.labels_:
  seg_img.append(kmeans.cluster_centers_[label])

seg_img = np.array(seg_img)
seg_img = seg_img.astype(int)
seg_img = seg_img.reshape(image.shape)

plt.imshow(seg_img)
print(seg_img.shape)

#Display the images
plt.subplot(121)
plt.title('Original Image')
plt.imshow(image)

plt.subplot(122)
plt.title('Segmented 4 colors')
plt.imshow(seg_img)
plt.show()

