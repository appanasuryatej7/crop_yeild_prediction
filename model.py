import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode categorical columns
le_soil = LabelEncoder()
le_crop = LabelEncoder()

data["SoilType"] = le_soil.fit_transform(data["SoilType"])
data["Crop"] = le_crop.fit_transform(data["Crop"])

# Features & Target
X = data.drop("Yield", axis=1)
y = data["Yield"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "crop_model.pkl")
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")

print("Model trained successfully!")
