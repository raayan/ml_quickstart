## Introduction
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

## Understanding the Data
"""
# Shape
print("Shape")
print(dataset.shape)

# Head
print("Head")
print(dataset.head(20))

# Descriptions
print("Descriptions")
print(dataset.describe())

# Class Distribution
print("Class Distribution")
print(dataset.groupby('class').size())

# Box and Whisker Plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Basic Histogram Plot
dataset.hist()
plt.show()

# Multivariate Plot
# Shows relationship between each combination of attributes (useful for spotting relationships)
scatter_matrix(dataset)
plt.show()
"""

## Modeling and Validation
# Split set for train/test
arr = dataset.values
X = arr[:,0:4] # Take only the attributes for prediction (not class)
Y = arr[:,4] # Take only the classes for validation
validation_size = 0.20 # 80/20 train/test
seed = 7 # Random gen

# Generate Train/Test
"""
Notes:
*X_train* will be the 80% of the data set with only the attributes (sepal_width, sepal_height, petal_width, petal_height)
*X_validation* will be the 20% of the data set with only the attributes (sepal_width, sepal_height, petal_width, petal_height)
*Y_train* will be the 80% of the data set with only the class [Iris-{setosa,versicolor,virginica}]
*Y_validation* will be the 20% of the data set with only the class [Iris-{setosa,versicolor,virginica}]

The *_train* part will be used to teach the model
the *_validation* part will be used to validate the models performance
"""
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test Options and Evaluation Metric
seed = 7 # Random gen
scoring = 'accuracy'

# Models
"""
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""

## Prediction
# Make predictions on validation dataset
# We'll just the KNN algorithm since it is simple and accurate
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

# Print Results
print("Accuracy:\n%f" % accuracy_score(Y_validation, predictions))
print("Confusion Matrix:\n%s" % confusion_matrix(Y_validation, predictions))
print("Classification Report:\n%s" % classification_report(Y_validation, predictions))

## Predicting a suggestion with a trained model 
# Enter inputs
idx = 0
test = X_validation[idx]
actual = Y_validation[idx]
print("Using data:\t\t%s" % test)
print("Actual class:\t\t%s" % actual)

# Predict based on inputs
prediction = knn.predict([test])[0]
print("Predicted class:\t%s" % prediction)
print("Prediction correct:\t%s" % (prediction == actual))
