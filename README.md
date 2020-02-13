# Hackerearth-Challenge
HackerEarth Machine Learning challenge: Calculating the severity of an airplane accident.
## The Challenge

>Flying has been the go-to mode of travel for years now; it is time-saving, affordable, and extremely convenient. According to the FAA, 2,781,971 passengers fly every day in the US, as in June 2019. Passengers reckon that flying is very safe, considering strict inspections are conducted and security measures are taken to avoid and/or mitigate any mishappenings. However, there remain a few chances of unfortunate incidents.
Imagine you have been hired by a leading airline. You are required to build Machine Learning models to anticipate and classify the severity of any airplane accident based on past incidents. With this, all airlines, even the entire aviation industry, can predict the severity of airplane accidents caused due to various factors and, correspondingly, have a plan of action to minimize the risk associated with them.

## The Data
The dataset comprises 3 files: 

* Train.csv: [10000 x 12 excluding the headers] contains Training data
* Test.csv: [2500 x 11 excluding the headers] contains Test data
* sample_submission.csv: contains a sample of the format in which the Results.csv needs to be

|Columns|Description|
|-----|------------------------|
|Accident_ID	|unique id assigned to each row
|Accident_Type_Code	|the type of accident (factor, not numeric)|
|Cabin_Temperature	|the last recorded temperature before the incident, measured in degrees fahrenheit|
|Turbulence_In_gforces	|the recorded/estimated turbulence experienced during the accident|
|Control_Metric	|an estimation of how much control the pilot had during the incident given the factors at play|
|Total_Safety_Complaints	|number of complaints from mechanics prior to the accident|
|Days_Since_Inspection	|how long the plane went without inspection before the incident|
|Safety_Score	|a measure of how safe the plane was deemed to be|
|Violations	|number of violations that the aircraft received during inspections|
|Severity	| a description (4 level factor) on the severity of the crash [Target]|

The data can be found in the link of the [HackerEarth Challenge](https://www.hackerearth.com/challenges/competitive/airplane-accident-severity-hackerearth-machine-learning-challenge/machine-learning/how-severe-can-an-airplane-accident-be-03e7a3f1/)

## Approach to solve the problem

On having the dataset, the first thing to do is analyse the data.
So, Exploratory data Analysis(EDA) was carried out on the training data. It was observed that the columns-
* Accident ID
* Turbulence_In_gforces
* Adverse_Weather_Metric
* Max_Elevation

Had very less impact on the target variable we are trying to predict.
This was done by using either the correlation matrix plot or Kernel density estimate (kde).
 
#### The Algorithm
The Algo proved to make most accurate prediction was *XGBoost Classifier*!
Other ML algorithms, such as
* Random Forest Classifier
* Support Vector Machine
* Deep Neural Networks
* CatBoost Classifier

were also deployed on this dataset.

### The Scikit-Learn code
The Scikit Learn code for the implementation of the above algorithms are:
##### XGBoost Classifier
```
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=.7,max_depth=20)
model.fit(x,y)
```
##### RandomForestClassifier
```
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 250, criterion = "entropy")
classifier.fit(x_train,y_train)
```
##### CatBoost Classifier
```
from catboost import CatBoostClassifier
import catboost as cb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [300]}
cb = cb.CatBoostClassifier()

modelC = GridSearchCV(cb, params)
modelC.fit(x_train,y_train)
```
#### Neural Network
```
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(7,15), activation='logistic', solver='adam',max_iter=1500, alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, shuffle=True, random_state=42)
```
#### SVM
```
from sklearn import svm
clf = svm.SVC(C=1.0, break_ties=False, cache_size=20000, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(x_train, y_train)
```

### The Results

|Algorithm|Accuracy on the Test Set|
|-----------|-----|
|XGBoost Classifier|95.51%|
|CatBoost Classifier|95.12%|
|Random Forest Classifier|94.64%|
|Deep Neural Networks|70.20%|
|Support Vector Machine|30.32%|

The final **_XGBoost Classifier_** used gave an F1-Score of **86.31167 with rank _8th_ on submission**.

**Thank You for reading!!**

Connect with me on

* [LinkedIN](in.com/in/debashish-nanda-b38164160/)
* [Hackerrank](https://www.hackerrank.com/debashish_nanda1)
