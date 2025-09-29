#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
#Reading the data in a csv file using pandas
df = pd.read_csv('email_classification.csv')
#print(df)
df.iloc[:, -1] = df.iloc[:, -1].str.strip().str.lower().map({'ham': 0.0, 'spam': 1.0})
#print(df)

X=[]
Y=[]
class_predicted = []
num_errors = 0
#Loop your data to allow each instance to be your test set
for i in df.index:
    pass
    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = df.drop(i).iloc[:, :-1].astype(float).values.tolist()

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = df.drop(i).iloc[:, -1].values.tolist()

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    test_sample = df.iloc[i, :-1].astype(float).values.tolist()

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([test_sample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted == df.iloc[i, -1]:
        num_errors +=1
#print(X)
#print(Y)
#Print the error rate
#--> add your Python code here
print("Error rate:", num_errors/len(df))