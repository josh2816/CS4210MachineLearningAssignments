#-------------------------------------------------------------------------
# AUTHOR: Joshua Yi
# FILENAME: decision_tree.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
#print(db)
#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
categorized_features = {
   'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, #Age
    'Myope': 1, 'Hypermetrope': 2, #Spectacle Prescription
    'No': 1, 'Yes': 2, #Astigmatism
    'Normal': 1, 'Reduced': 2 #Tear Production Rate
  }
# X =
for row in db:
    encoded_row = []
    for i in range(4):
        encoded_row.append(categorized_features[row[i]])
    X.append(encoded_row)
#print(db)
#print("\n")
#print(X)
#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
categorized_classes = {'Yes': 1, 'No': 2}
# Y =
for row in db:
    Y.append(categorized_classes[row[4]])
#print(Y)
#fitting the decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
clf = tree.DecisionTreeClassifier(criterion='entropy')
#clf =
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()