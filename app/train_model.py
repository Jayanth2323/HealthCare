# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/cleaned_blood_data.csv")

# Define features and target
X = df[[
    "Age", "Gender", "Blood_Pressure", "Heart_Rate",
    "Cholesterol", "Hemoglobin", "Smoking_Status", "Exercise_Level"
]]
y = df["Class"]

# Preprocessing
numeric_features = ["Age", "Blood_Pressure", "Heart_Rate", "Cholesterol", "Hemoglobin"]
categorical_features = ["Gender", "Smoking_Status", "Exercise_Level"]

numeric_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("encoder", OneHotEncoder(drop="first"))])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Final pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model_pipeline, "models/logistic_regression_pipeline.pkl")
print("✅ Model saved to models/logistic_regression_pipeline.pkl")
