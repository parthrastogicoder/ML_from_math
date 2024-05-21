from sklearn import datasets
from sklearn.model_selection import train_test_split
from Decision_tree import DecisionTree
from Decision_tree import Node
from randomforrest import RandomForest
from Adaboost import Adaboost
from Adaboost import DecisionStump
from perceptron import Perceptron
from supportvectormachine import SVM
import numpy as np


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy: DT :", acc)
cl = RandomForest(n_trees=10, max_depth=10)

cl.fit(X_train, y_train)
y_pred = cl.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy: RF :", acc)
for i in range(10,100):
    c = Adaboost(n_clf=i)
    c.fit(X_train, y_train)
    y_pred = c.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("num of decision stump",i)
    print("Accuracy:", acc)

p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

# print("Perceptron classification accuracy", accuracy(y_test, predictions))
p = SVM(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Accuracy : SVM:", accuracy(y_test, predictions))