# Udacity MLND Capstone Project

####
# This code takes approximately 30 minutes to run on an intel i7-4770 CPU with 20GB ram
####

# Importing the libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

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

# Obtain the F-score of every principal component from n_components = 2 to 23
for i in range(2, 23): 
    print("{} principal components".format(i))
    # Perform Feature Extraction using PCA
    pca = PCA(n_components = i)
    pca_values = pca.fit(X_train)

    # Take Principal components of X_train and throw into classifier
    pc_train = pca.transform(X_train)
    pc_test = pca.transform(X_test)
    
    # Fitting RBF SVM to the Training set
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(pc_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(pc_test)

    # Calculate F1 score of RBF Kernel SVC
    score = f1_score(y_test, y_pred)
    print("RBF SVC score: {}".format(score))
    
    # Performing Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators = 500, criterion = "gini", random_state = 0)
    classifier.fit(pc_train, y_train)    
    
    # Predicting the Test set results
    y_pred = classifier.predict(pc_test)

    # Calculate F1 score of Random Forest Classifier
    score = f1_score(y_test, y_pred)
    print("Random Forest score: {}".format(score))    
    
    # Performing Adaboost
    classifier = AdaBoostClassifier(n_estimators = 500, random_state = 0)
    classifier.fit(pc_train, y_train)    
    
    # Predicting the Test set results
    y_pred = classifier.predict(pc_test)

    # Calculate F1 score of Adaboost
    score = f1_score(y_test, y_pred)
    print("AdaBoost score: {}".format(score))        
    
    print("----------------------------------")
    
'''
No PCA
RBF SVC score: 0.4457284358233594'

16 principal components
RBF SVC score: 0.4520098441345365
Random Forest score: 0.44960629921259837
AdaBoost score: 0.4307944307944308

17 principal components
RBF SVC score: 0.45002056766762644
Random Forest score: 0.4501582278481013
AdaBoost score: 0.42898076135898494

The most optimal number of principal components in this case is either 16/17. The RBF SVC has the highest
F-score of 0.452 when there are 16 principal components. But, the Random Forest Classifier comes in really
close to the RBF SVC F-score with a score of 0.45 when there are 17 principal components. With such a slight
difference in the scoring, it's safe to say that both the RBF SVC and Random Forest Classifier will work well
for this dataset. In comparison to the previous benchmark model which was a RBF SVC without using principal 
components, there is a slight increase in the F-score. Withouting using any principal components, we yield
a F-score of 0.446. That's only approximately a 0.01 increase. 
'''

# Summarize components for 16 components
pca = PCA(n_components = 16)
pca_values = pca.fit(X_train)

print("Explained Variance: {}".format(pca_values.explained_variance_ratio_))
print(pca_values.components_)
components = pca_values.components_
explained_components = pca_values.explained_variance_
explained_components_ratio = pca_values.explained_variance_ratio_