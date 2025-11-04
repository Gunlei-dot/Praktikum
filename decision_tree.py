import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split #function currently not used
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # metrics for manual model eval

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

X_train = X[X.set_train == True]
X_test = X[X.set_val == True]
y_train = y[y.set_train == True]
y_test = y[y.set_val == True]

# Dropping the set indicator columns after the split
X_train = X_train.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
X_test = X_test.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
y_test = y_test.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
y_train = y_train.drop(columns=['set_train', 'set_val']) # Drop the set indicator columns
y_test = np.array(y_test).astype(str) # Convert y_test to a NumPy array of strings
y_train = np.array(y_train).astype(str) # Convert y_train to a NumPy array of strings
#y_train = y_train.flatten() # flatten both as sklearn expects 1D array for y
#y_test = y_test.flatten()

with mlflow.start_run() as run:  # Everything inside this block is logged
    print("✅ MLflow run started successfully!")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")   
    
    forest_params = {
        "min_weight_fraction_leaf": 0.1,
        "n_estimators": 20,
    }

    # Log parameters
    mlflow.log_params(forest_params)

    # Train model
    model = RandomForestClassifier(**forest_params)
    model = model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Log metrics and model
    mlflow.log_metrics(metrics)
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="Random_forest_model",
        signature=signature
    )


print("Metrics logged to MLflow:")
print("Classification Report:\n", classification_report(y_test, y_pred))
