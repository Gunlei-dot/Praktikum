import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split #function currently not used
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # metrics for manual model eval
from sklearn.model_selection import cross_val_score


'mlflow.sklearn.autolog()'

patient_data = pd.read_csv(r'C:\Users\s434037\Desktop\Bachelor\projects\labels.tsv', encoding='utf-8', sep='\t') #encoding and sep to read tsv correctly
patient_data = patient_data.dropna() # Drop rows with missing values for simplicity 
patient_data = patient_data.drop(columns=['pseudo_id', 'sex', 'pseudo_patid']) # Drop patient_id as it's not a feature for prediction
patient_data = patient_data[patient_data.label != 2] # Remove rows with label 2 as these are not relevant for binary classification
patient_data = patient_data[patient_data.psa != 'NA'] # remove rows with no psa value till i find a better solution
patient_data = patient_data[patient_data.staging != 'primary'] # remove rows with primary staging till i find a better solution

patient_data['age'] = patient_data['age'].astype(float) # convert psa to float
patient_data['px'] = patient_data['px'].astype(float) # convert psa to float

train_mask = patient_data['set'] == 'train' 
test_mask = patient_data['set'] == 'val' # splitting train/val before one-hot encoding

X = pd.get_dummies(patient_data.drop("label", axis=1)) # dummies for categorical variables since forest doesn't handle them directly
X.index = patient_data.index
y = patient_data[["label"]].astype(int) # Keeping y as DataFrame for easier handling of set indicators

'''forest_params = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": None,
    "random_state": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,
    "min_weight_fraction_leaf": 0.1,
                }


tree_test = tree.DecisionTreeClassifier(**tree_params)

scores = cross_val_score(tree_test, X, y, cv=5)
print(f"Cross-validation scores: {scores}")''' # section used for testing cross validation before implementing grid search

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

X_train = X_train.drop(columns=['set_train', 'set_val'], errors= 'ignore') # Drop the set indicator columns
X_test = X_test.drop(columns=['set_train', 'set_val'], errors = 'ignore')

y_test = np.array(y_test).astype(int) # Convert y_test to a NumPy array of strings
y_train = np.array(y_train).astype(int) # Convert y_train to a NumPy array of strings

y_test = y_test.squeeze()
y_train = y_train.squeeze() # sections sets up train test split and ensures proper data types

param_grid = {
    'min_samples_split': [2, 3, 4],
    'min_weight_fraction_leaf': [0.1],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'max_leaf_nodes': [None, 10, 20],
}

# Dropping the set indicator columns after the split

with mlflow.start_run(run_name='random_forest_param_tuning'): # Start an MLflow run to log parameters, metrics, and the model
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1', error_score='raise')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    model = RandomForestClassifier(**best_params)
    model = model.fit(X_train, y_train)

    
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
print(f"Test score: {best_score:.3f}")
