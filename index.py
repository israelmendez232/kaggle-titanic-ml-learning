import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sys
import os

columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Preparing the data.
trainData = pd.read_csv("./data/train.csv", sep=',', quotechar="\"")

df = pd.DataFrame(trainData, columns = columns)

y = trainData.Survived
x = trainData[columns]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Creating and testing the model.
model1 = DecisionTreeRegressor(random_state=1)
model2 = RandomForestClassifier(random_state=1)

model1.fit(train_x, train_y)
model2.fit(train_x, train_y)

# Test with the final test.csv
validation_pre = model1.predict(val_x)
validation_mae = mean_absolute_error(validation_pre, val_y)

print(validation_mae)

# Print the output:


