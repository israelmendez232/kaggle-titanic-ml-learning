import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import csv
import numpy as np
import sys
import os

columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
columns1 = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Preparing the data for Train and Test.
trainData = pd.read_csv("./data/train.csv", sep=',')
dfTrain = pd.DataFrame(trainData, columns = columns)

testData = pd.read_csv("./data/test.csv", sep=',')
dfTest = pd.DataFrame(testData, columns = columns1)

# Encoding for the Train and Test data.
le = preprocessing.LabelEncoder()
for i in columns:
     dfTrain[i] = le.fit_transform(dfTrain[i].astype('str'))

for i in columns1:
     dfTest[i] = le.fit_transform(dfTest[i].astype('str'))

y = dfTrain.Survived
x = dfTrain[columns]

val_xT = dfTest[columns1]

train_x, val_x, train_x, val_y = train_test_split(x, y, random_state=1)

# Create and predicts with the model.
model1 = DecisionTreeRegressor(random_state=1)
model1.fit(val_x, val_y)

prediction = model1.predict(val_xT)

# Print the output:
output = pd.DataFrame({"PassengerId":  dfTest.PassengerId, "Survived": prediction})
output.to_csv("./data/test.csv", sep=',', index=False)


""" ----> CODE TO VALIDATE THE MODEL <----
validation_pre1 = model1.predict(val_x)
print(val_x)
validation_mae1 = mean_absolute_error(validation_pre1, val_y)

validation_pre2 = model2.predict(val_x)
validation_mae2 = mean_absolute_error(validation_pre2, val_y)
"""
