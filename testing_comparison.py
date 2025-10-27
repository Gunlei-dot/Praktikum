import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn import tree
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split #function currently not used
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # metrics for manual model eval

mlflow.sklearn.autolog(max_tuning_runs=10)

patient_data = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\projects\labels.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly
patient_data = patient_data.dropna() # Drop rows with missing values for simplicity 
patient_data = patient_data.drop(columns=['pseudo_id']) # Drop patient_id as it's not a feature for prediction
patient_data = patient_data.drop(columns=['sex']) # Drop sex as it's not a feature for prediction
patient_data = patient_data.drop(columns=['pseudo_patid']) # Drop pst_id as it's not a feature for prediction
patient_data = patient_data[patient_data.label != 2] # Remove rows with label 2 as these are not relevant for binary classification
patient_data = patient_data[patient_data.psa != 'NA'] # remove rows with no psa value till i find a better solution
patient_data = patient_data[patient_data.staging != 'primary'] # remove rows with primary staging till i find a better solution
patient_data['age'] = patient_data['age'].astype(float) # convert psa to float
patient_data['px'] = patient_data['px'].astype(float) # convert psa to float

X = pd.get_dummies(patient_data.drop("label", axis=1)) # dummies for categorical variables since DecisionTree doesn't handle them directly
y = pd.get_dummies(patient_data.drop(columns=["age", "staging", "px", "psa"])) # Features and target variable

#X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)  (#random_state for reproducibility, same split)

X_train =   X[X.set_train == True]
X_test =    X[X.set_val == True]
y_train =   y[y.set_train == True]
y_test =    y[y.set_val == True]

param_grid = {
    'criterion': ['log_loss'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'ccp_alpha': [0.0, 0.01, 0.1]
}

# Dropping the set indicator columns after the split
X_train = X_train.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
X_test = X_test.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
y_test = y_test.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
y_train = y_train.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
y_test = np.array(y_test).astype(str) # Convert y_test to a NumPy array of strings
y_train = np.array(y_train).astype(str) # Convert y_train to a NumPy

with mlflow.start_run(run_name='decision_tree_param_tuning'): # Start an MLflow run to log parameters, metrics, and the model
    dt_classifier = tree.DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    model = tree.DecisionTreeClassifier(**best_params)
    model = model.fit(X_train, y_train)

    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Test score: {best_score:.3f}")
