import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Truth table for AND logic gate
data = np.array([[0,0], [1,0], [0,1], [1,1]])
and_labels = np.array([0, 0, 0, 1])
# Truth table for OR logic gate
or_labels = np.array([0, 1, 1, 1])
# Truth table for XOR logic gate
xor_labels = np.array([0, 1, 1, 0])

labels = xor_labels
# plot points
plt.figure()
plt.scatter([x[0] for x in data], and_labels, c=and_labels)
plt.show()
plt.clf()

# Instantiate perceptron and train
classifier = Perceptron(max_iter = 40, random_state=22)
classifier.fit(data, labels)
print(f'Perceptron classifier score: {classifier.score(data, labels)}')

# Visualizing the perceptron
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))

abs_distances = abs(classifier.decision_function(point_grid))
abs_distances_matrix = np.reshape(abs_distances, (100, 100))

heat_map = plt.pcolormesh(x_values, y_values, abs_distances_matrix)
plt.colorbar(heat_map)
plt.show()
