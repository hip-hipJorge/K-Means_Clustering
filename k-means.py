import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from sklearn import datasets
from sklearn.metrics import classification_report, mean_squared_error

# load in iris data
iris = datasets.load_iris()
X = scale(iris.data)
# target
y = pd.DataFrame(iris.target)

variable_names = iris.feature_names

# Building and running the model
clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(X)

# Plotting your model outputs
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal Length', 'Sepal_width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth classification')


plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)

plt.title('K-Means Classification')
plt.show()

relabel = np.choose(clustering.labels_,[2,0,1]).astype(np.int64)

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Relabeld Ground Truth classification')


plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)

plt.title('Relabeled K-Means Classification')
plt.show()

# Evaluate the cluster results
print(classification_report(y, relabel))
print(mean_squared_error(y, relabel))
