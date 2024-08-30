import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

# Load dataset
hf = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Change Quality to Category
hf['DEATH_EVENT'] = hf['DEATH_EVENT'].astype('category')

# Set Features and Target Variable (DEATH_EVENT)
X = hf[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']].copy()
y = hf[['DEATH_EVENT']].copy()

# Split data for testing and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# Build Model
hfModel = KNeighborsClassifier(n_neighbors=5)

# Train Model
hfModel.fit(X_train, y_train)

# Make Predictions
predictions = hfModel.predict(X_test)

# Pickle Model
import pickle
pickle.dump(hfModel, open('knn.pkl','wb'))