import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 🔹 Load dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")

# 🔹 Handle missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 🔹 Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 🔹 Set target and features
target = "bank_account"
X = df.drop(columns=[target])
joblib.dump(X.columns.tolist(), "feature_names.pkl")

y = df[target]

# 🔹 Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 🔹 Save the model, scaler, and encoders
joblib.dump(model, "financial_inclusion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# 🔹 Optional Evaluation
y_pred = model.predict(X_test)
print("✅ Model trained and saved successfully!")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
