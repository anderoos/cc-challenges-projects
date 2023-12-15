import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load in digits dataset
digits = datasets.load_digits()

# Describe dataset
# print(digits.DESCR)
# print(digits.data)
print(digits.target)

# Preview images
plt.gray()
plt.matshow(digits.images[100])
plt.show()
plt.close()

print(digits.target[100])

# Build K means cluster data
# Get number of clusters
values = len(np.unique(digits.target))

model = KMeans(n_clusters = 10, random_state=3)
model.fit(digits.data)

# Build plot to visualize centroids
fig = plt.figure(figsize=(8,3))
fig.suptitle("Cluster Center Images")

for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)\
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,5.03,2.59,0.00,0.00,0.00,0.00,0.00,0.00,6.86,3.81,0.00,0.00,0.00,0.00,0.00,0.00,6.86,3.81,0.00,0.00,0.00,0.00,0.00,0.00,6.86,3.81,0.00,0.00,0.00,0.00,0.00,0.00,6.86,3.81,0.00,0.00,0.00,0.00,0.00,0.00,6.79,3.74,0.00,0.00,0.00,0.00,0.00,0.00,0.99,0.38,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.53,4.42,0.53,0.00,0.00,0.53,7.40,1.83,1.52,7.62,1.52,0.00,0.00,0.76,7.62,2.29,1.52,7.62,1.52,0.00,0.00,0.91,7.62,2.29,2.90,7.62,1.52,0.00,0.00,1.52,7.62,7.25,7.62,7.62,1.52,0.00,0.00,1.30,7.63,6.10,4.05,7.62,1.52,0.00,0.00,0.00,0.69,0.00,1.14,7.47,1.14,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,5.03,6.10,6.10,6.10,6.10,6.10,5.04,0.00,3.66,4.57,4.57,4.57,4.57,6.87,6.56,0.00,0.00,0.00,0.00,0.00,0.00,6.49,5.12,0.00,0.00,0.00,0.23,2.29,2.51,7.55,3.81,0.00,0.00,0.00,2.06,7.62,7.62,7.62,2.06,0.00,0.00,0.00,0.00,0.76,3.36,7.62,0.92,0.00,0.00,0.00,0.00,0.00,4.58,7.09,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.08,2.21,2.29,0.69,0.00,0.00,0.00,0.00,1.30,7.63,7.62,5.34,0.00,0.00,0.00,0.00,0.00,0.69,4.65,6.86,0.00,0.00,0.00,0.00,0.00,0.00,3.81,6.86,0.00,0.00,0.00,0.00,0.00,0.69,4.96,7.24,5.18,4.88,0.00,0.00,1.37,7.24,7.62,7.62,6.79,6.41,0.00,0.00,1.15,6.94,7.62,5.41,0.08,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
