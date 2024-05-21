

# Machine Learning Algorithms Repository

This repository contains implementations of several machine learning algorithms including PCA, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Logistic Regression (LR), Decision Tree, Random Forest, AdaBoost, and Perceptron. Additionally, classes for testing these implementations are also included.

## Algorithms Implemented

1. **Principal Component Analysis (PCA)**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Logistic Regression (LR)**
5. **Decision Tree**
6. **Random Forest**
7. **AdaBoost**
8. **Perceptron**

## Testing Classes

For each algorithm, corresponding classes are provided to facilitate testing and evaluation. These classes include methods to train the models, make predictions, and assess performance using appropriate metrics.

## Installation

To use this repository, clone it to your local machine 
Make sure you have the necessary Libraries installed: like scikit learn , numpy , collections. 

## Usage

Below is a brief overview of how to use the classes for a couple of algorithms. For detailed usage, refer to the individual files.

### Example: Using PCA

```python
from pca import PCA

# Assuming X is your dataset
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)
```

### Example: Using KNN and other classifiers , regressor 

```python
from knn import KNN

# Assuming X_train, y_train, X_test are your datasets
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```



## Contributing

If you wish to contribute to this repository, please fork the repository and submit a pull request. For significant changes, please open an issue first to discuss what you would like to change.

