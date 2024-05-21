import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from pca import PCA
data = datasets.load_iris()
X = data.data
y = data.target
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of trans X:", X_projected.shape)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
plt.xlabel("Pc 1")
plt.ylabel("Pc 2")
plt.colorbar()
plt.show()