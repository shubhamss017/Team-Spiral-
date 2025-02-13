import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Crop_recommendation_with_sandy_texture.csv")

# Encode categorical columns
label_encoders = {}
for col in ["label", "Primary_Fertilizer", "Soil_Texture"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for decoding

# Define features and targets
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "Soil_Texture"]
X = df[features]

y_crop = df["label"]  # Crop prediction target
y_fertilizer = df["Primary_Fertilizer"]  # Fertilizer prediction target

# Split data into training and testing sets
X_train, X_test, y_crop_train, y_crop_test = train_test_split(X, y_crop, test_size=0.2, random_state=42)
X_train_fert, X_test_fert, y_fert_train, y_fert_test = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)

# Apply SMOTE to balance fertilizer data
smote = SMOTE(random_state=42)
X_train_fert_balanced, y_fert_train_balanced = smote.fit_resample(X_train_fert, y_fert_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_fert_scaled = scaler.transform(X_train_fert_balanced)
X_test_fert_scaled = scaler.transform(X_test_fert)

# Train Crop Recommendation Model using Random Forest
crop_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
crop_model.fit(X_train_scaled, y_crop_train)
crop_preds = crop_model.predict(X_test_scaled)
crop_acc = accuracy_score(y_crop_test, crop_preds)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_tuned = RandomizedSearchCV(
    XGBClassifier(random_state=42, tree_method='hist'),
    param_grid,
    cv=5,
    n_iter=10,
    scoring='accuracy'
)

xgb_tuned.fit(X_train_fert_scaled, y_fert_train_balanced)
fertilizer_model = xgb_tuned.best_estimator_

fert_preds = fertilizer_model.predict(X_test_fert_scaled)
fert_acc = accuracy_score(y_fert_test, fert_preds)

# Save models and encoders
joblib.dump(crop_model, "crop_recommendation_model.pkl")
joblib.dump(fertilizer_model, "fertilizer_recommendation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print(f"Crop Model Accuracy: {crop_acc * 100:.2f}%")
print(f"Fertilizer Model Accuracy: {fert_acc * 100:.2f}%")

# User Input for Soil Conditions
print("\nPlease enter the following soil conditions:")
user_input = {
    "N": float(input("Enter Nitrogen value: ")),
    "P": float(input("Enter Phosphorus value: ")),
    "K": float(input("Enter Potassium value: ")),
    "temperature": float(input("Enter Temperature: ")),
    "humidity": float(input("Enter Humidity: ")),
    "ph": float(input("Enter pH value: ")),
    "rainfall": float(input("Enter Rainfall: ")),
    "Soil_Texture": input("Enter Soil Texture (Clay, Loam, Sandy): ")
}

# Encode soil texture
if user_input["Soil_Texture"] in label_encoders["Soil_Texture"].classes_:
    user_input["Soil_Texture"] = label_encoders["Soil_Texture"].transform([user_input["Soil_Texture"]])[0]
else:
    print("Invalid Soil Texture")
    exit()

# Convert input to DataFrame
user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)

# Predict Crop and Fertilizer
crop_prediction = crop_model.predict(user_df_scaled)
fertilizer_prediction = fertilizer_model.predict(user_df_scaled)

# Decode predictions
crop_name = label_encoders["label"].inverse_transform(crop_prediction)[0]
fertilizer_name = label_encoders["Primary_Fertilizer"].inverse_transform(fertilizer_prediction)[0]

print(f"\nRecommended Crop: {crop_name}")
print(f"Recommended Fertilizer: {fertilizer_name}")

# Visualization
labels = ['Crop Model Accuracy', 'Fertilizer Model Accuracy']
accuracy = [crop_acc * 100, fert_acc * 100]
plt.figure(figsize=(6, 4))
plt.bar(labels, accuracy, color=['green', 'blue'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 100])
plt.show()
