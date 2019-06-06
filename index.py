from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import tree
from pandas import Series, DataFrame
import pandas as pd
import csv
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
     trainData[i] = le.fit_transform(trainData[i].astype('str'))

for i in columns1:
     testData[i] = le.fit_transform(testData[i].astype('str'))

y = trainData.Survived
X = trainData[columns1]
val_xT = testData[columns1]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Create and predicts with the model.
model1 = DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)

prediction = model1.predict(val_xT)

# Print the output:
output = pd.DataFrame({"PassengerId":  dfTest.PassengerId, "Survived": prediction})
output.to_csv("./data/output.csv", sep=',', index=False)

""" 
----> CODE TO VALIDATE THE MODEL <----
validation_pre1 = model1.predict(val_x)
print(val_x)
validation_mae1 = mean_absolute_error(validation_pre1, val_y)

validation_pre2 = model2.predict(val_x)
validation_mae2 = mean_absolute_error(validation_pre2, val_y)

 ----> CODE TO VALIDATE THE MODEL (Part. II) <----
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

y_test_size = y_test.size
print(y_test_size)
y_train_size = y_train.size
print(y_test_size)
X_train = [X_train]
y_train = [y_train]
X_test = [X_test]
y_test = [y_test]

c = tree.DecisionTreeClassifier()
c.fit(X_train.values.reshape(-1, 1), y_train)

accu_train = np.sum(c.predict(X_train.values.reshape(-1, 1)) == y_train)/y_train_size
accu_test = np.sum(c.predict(X_test.values.reshape(-1, 1)) == y_test)/y_test_size

print("Accuracy on Train: ", (accu_train * 100))
print("Accuracy on Test: ", (accu_test * 100))

# Removing .0 from prediction
# removing0 = [i.rstrip('.') for i in str(prediction)]
"""
