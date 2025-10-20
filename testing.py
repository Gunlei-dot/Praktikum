import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import mlflow
#import mlflow.sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#mlflow.sklearn.autolog() # Enable automatic logging for sklearn

patient_data = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\projects\labels.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly
patient_data = patient_data.dropna() # Drop rows with missing values for simplicity 
patient_data = patient_data.drop(columns=['pseudo_id']) # Drop patient_id as it's not a feature for prediction
patient_data = patient_data.drop(columns=['sex']) # Drop sex as it's not a feature for prediction
patient_data = patient_data.drop(columns=['pseudo_patid']) # Drop pst_id as it's not a feature for prediction
patient_data = patient_data[patient_data.label != 2] # Remove rows with label 2 as these are not relevant for binary classification

#set_train = [patient_data(set)]

X = pd.get_dummies(patient_data.drop("label", axis=1)) # dummies for categorical variables since DecisionTree doesn't handle them directly
y = pd.get_dummies(patient_data.drop(columns=["age", "staging", "px", "psa"])) # Features and target variable
#X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)  (#random_state for reproducibility, same split)

X_train =   X[X.set_train == True]
X_test =    X[X.set_val == True]
y_train =   y[y.set_train == True]
y_test =    y[y.set_val == True]

y_test = np.array(y_test).astype(str) # Convert y_test to a NumPy array of strings
y_train = np.array(y_train).astype(str) # Convert y_train to a NumPy

#with mlflow.start_run(): # Start an MLflow run to log parameters, metrics, and the model
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)
predictions = model.predict(X_test)

y_test=np.array([row[0] for row in y_test])
predictions=np.array([row[0] for row in predictions])
accuracy = accuracy_score(y_test, predictions) # Calculate accuracy

#print(f"Training accuracy: {train_score:.3f}")
#print(f"Test accuracy: {test_score:.3f}")

print("Accuracy:", accuracy)
print(confusion_matrix(y_test, predictions))   
print(classification_report(y_test, predictions))
  
#plt.figure(figsize=(12,12)) # setting the figure size
#tree.plot_tree(model, filled=True, fontsize=6) # Visualize the decision tree 
#plt.show() # Display the decision tree
