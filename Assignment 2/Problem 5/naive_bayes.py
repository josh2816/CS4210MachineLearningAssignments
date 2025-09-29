#-------------------------------------------------------------------------
# AUTHOR: Joshua Yi
# FILENAME: naive_bayes.py
# SPECIFICATION: executes the naive bayes algorithm to classify weather data
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
categorized_features = {
   'Sunny': 1, 'Overcast': 2, 'Rain': 3, #Weather
    'Hot': 1, 'Cool': 2, 'Mild': 3, #Temperature
    'High': 1, 'Normal': 2, #Humidity
    'Weak': 1, 'Strong': 2 #Windy
  }
X = []
for row in dbTraining:
    encoded_row = []
    for i in range(1, 5):
        encoded_row.append(categorized_features[row[i]])
    X.append(encoded_row)
#print(X)
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
categorized_classes = {'Yes': 1, 'No': 2}
Y = []
for row in dbTraining:
    encoded_row = []
    for i in range(5, 6):
        encoded_row.append(categorized_classes[row[i]])
    Y.append(encoded_row)
#print(Y)
#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf = clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header of the solution
#--> add your Python code here
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")
#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    test_sample = []
    for i in range(1, 5):
        test_sample.append(categorized_features[row[i]])
    prediction = clf.predict([test_sample])[0]
    confidence = max(clf.predict_proba([test_sample])[0])
    if confidence > 0.75:
        print(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {'Yes' if prediction == 1 else 'No'} {confidence:.2f}")