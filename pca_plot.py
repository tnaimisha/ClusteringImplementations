#pca_plot.py is a script using priciple component analysis technique that projects all the dimensions into 2 or 3 dimensional space as required. This is to get an idea about how the cluster looks like.

import pandas as pd
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import sys

url = sys.argv[1]
data = pd.read_csv(url)

y = data.ix[:,-1]          # Split off classifications
X = data.ix[:,:-1]
classes = list(y.unique())

X_norm = (X - X.min())/(X.max() - X.min())

pca = sklearnPCA(n_components=3) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

colors = ['red','blue','green','yellow','grey','orange','brown','black','violet','magenta','cyan','indigo','crimson','maroon','white']

for i in range(len(classes)): #print each class with a different color
    plt.scatter(transformed[y==classes[i]][0], transformed[y==classes[i]][1], label=classes[i], c=colors[i])

plt.legend()
plt.show()

