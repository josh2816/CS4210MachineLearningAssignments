#-------------------------------------------------------------------------
# AUTHOR: Joshua Yi
# FILENAME: decition_tree_2.py
# SPECIFICATION: encodes categorical features to numbers, creates a decision tree based on the three sets of
#training data, and tests it with the test data in a or loop of 10 runs to find the average accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())
print(df_test)
df_test[df_test.columns[0]] = df_test[df_test.columns[0]].map({
    'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3
})
df_test[df_test.columns[1]] = df_test[df_test.columns[1]].map({
    'Myope': 1, 'Hypermetrope': 2
})
df_test[df_test.columns[2]] = df_test[df_test.columns[2]].map({
    'No': 1, 'Yes': 2
})
df_test[df_test.columns[3]] = df_test[df_test.columns[3]].map({
    'Normal': 1, 'Reduced': 2
})
df_test[df_test.columns[4]] = df_test[df_test.columns[4]].map({
    'Yes': 1, 'No': 2
})
#print(df_test)
dbTest = df_test.values.tolist()
#print(dbTest)
for ds in dataSets:
    correct_predictions = 0
    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dbTraining = pd.read_csv(ds)
    
    #print(dbTraining)
    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    dbTraining[dbTraining.columns[0]] = dbTraining[dbTraining.columns[0]].map({
        'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3
    })
    dbTraining[dbTraining.columns[1]] = dbTraining[dbTraining.columns[1]].map({
        'Myope': 1, 'Hypermetrope': 2
    })
    dbTraining[dbTraining.columns[2]] = dbTraining[dbTraining.columns[2]].map({
        'No': 1, 'Yes': 2
    })
    dbTraining[dbTraining.columns[3]] = dbTraining[dbTraining.columns[3]].map({
        'Normal': 1, 'Reduced': 2
    })
    X = dbTraining.iloc[:, :-1].values.tolist()
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    dbTraining[dbTraining.columns[4]] = dbTraining[dbTraining.columns[4]].map({
        'Yes': 1, 'No': 2
    })
    Y = dbTraining.iloc[:, -1].values.tolist()
    
    #print(dbTraining)
    #Loop your training and test tasks 10 times here
    for i in range (10):
       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)
       tree.plot_tree(clf, 
                      feature_names=['Age', 'Spectacle Prescription', 'Astigmatism', 'Tear Production Rate'], 
                      class_names=['Yes','No'], 
                      filled=True, 
                      rounded=True)
       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           features = data[:-1]
           true_label = data[-1] 

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           predicted_label = clf.predict([features])[0]
           if predicted_label == true_label:
                correct_predictions += 1

    # Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    number_of_test_instances = len(dbTest)
    total_predictions = number_of_test_instances * 10
    average_accuracy = correct_predictions / total_predictions

    #Print the average accuracy of this model during the 10 runs (training and test set).
    print(average_accuracy)
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {average_accuracy:.1f}")
