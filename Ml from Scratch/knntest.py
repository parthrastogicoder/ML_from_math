import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
# print(X_train.shape)

from knn import KNN
cl = KNN(k=5)
cl.fit(X_train, y_train)
predictions=cl.predict(X_test)

acc = np.sum(predictions==y_test)/len(y_test)
print(acc)