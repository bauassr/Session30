

import numpy as np
from sklearn import cluster, datasets
from scipy import misc
import matplotlib.pyplot as plt


# Load Image(Data)


image = misc.face()
plt.imshow(image)


#Basic Information about the Image(Data)


image.shape


#Preprocess the Image into an Array




image_r = (image / 255.0).reshape(-1,3)




image_r.shape

# Use K-Means for 5 Clusters from Reshaped Image Array


from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=5).fit(image_r)
y_pred_kmeans=k_means.predict(image_r)



k_means.cluster_centers_.shape


k_means.labels_.shape


# Create New Image Array from the Generated Clusters and Labels  


newimg = k_means.cluster_centers_[k_means.labels_]


newimg.shape


#  Reshape the New Image Array with Original Image Dimensions 


newimg=np.reshape(newimg, (image.shape))



newimg.shape


# Plot Original and Color Compressed Image


fig = plt.figure(figsize=(8,8))
ax=fig.add_subplot(1,2,1,xticks=[],yticks=[],title='Original Image')
ax.imshow(image)
ax=fig.add_subplot(1,2,2,xticks=[],yticks=[],title='Color Compressed Image using K-Means')
ax.imshow(newimg)
plt.show()

