# Udacity MLND Capstone Project

# SVM Using a linear, rbf, polynomial, and sigmoid kernel

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('UCI_Credit_Card.csv')
X = dataset.iloc[:, 1:24].values
y = dataset.iloc[:, 24].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

############################################################
# Fitting Linear SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating the F1-score
from sklearn.metrics import f1_score
score = f1_score(y_test, y_pred)
print("Linear SVC score: {}".format(score))
############################################################
# Fitting RBF SVM to the Training set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating the F1-score
score = f1_score(y_test, y_pred)
print("RBF SVC score: {}".format(score))
############################################################
# Fitting Polynomial SVM to the Training set
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating the F1-score
score = f1_score(y_test, y_pred)
print("Polynomial SVC score: {}".format(score))
############################################################
# Fitting Sigmoid SVM to the Training set
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating the F1-score
score = f1_score(y_test, y_pred)
print("Sigmoid SVC score: {}".format(score))
############################################################

# F1-Scores obtained are as follows:
    
# Linear SVC score: 0.35892691951896394
# RBF SVC score: 0.4457284358233594
# Polynomial SVC score: 0.32003710575139144
# Sigmoid SVC score: 0.3055725658297612

# This shows that the model is most likely represented as a RBF kernel of the SVC.