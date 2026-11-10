
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

print("ZOMATO RESTAURANT RATING PREDICTION")
print("-----------------------------------")

# Load dataset
df = pd.read_csv("zomato_cleaned.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Total restaurants:", len(df))

# Select features and target
features = [
    "Average Cost for two",
    "Votes",
    "Price range",
    "Has Table booking",
    "Has Online delivery"
]

target = "Aggregate rating"

# Convert numerical columns
numeric_columns = [
    "Average Cost for two",
    "Votes",
    "Price range",
    "Aggregate rating"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Convert Yes/No columns into numbers
binary_columns = [
    "Has Table booking",
    "Has Online delivery"
]

for column in binary_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

# Prepare data
data = df[features + [target]].copy()

# Remove unrated and invalid target values
data = data[
    (data[target] > 0) &
    (data[target] <= 5)
]

# Handle missing feature values
for column in features:
    data[column] = data[column].fillna(data[column].median())

data = data.dropna()

X = data[features]
y = data[target]

print("Restaurants selected for prediction:", len(data))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale features for Linear Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models
linear_model = LinearRegression()

random_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train models
linear_model.fit(X_train_scaled, y_train)
random_model.fit(X_train, y_train)

print("Models trained successfully!")

# Make predictions
linear_predictions = linear_model.predict(X_test_scaled)
random_predictions = random_model.predict(X_test)

# Evaluate models
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 3)
    }

results = []

results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        linear_predictions
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        random_predictions
    )
)

# Save model comparison
results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE")
print(results_df.to_string(index=False))

# Actual vs Predicted graph
plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.4,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    random_predictions,
    alpha=0.4,
    label="Random Forest"
)

plt.plot(
    [0, 5],
    [0, 5],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Restaurant Rating")
plt.ylabel("Predicted Restaurant Rating")
plt.title("Actual vs Predicted Restaurant Ratings")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# Random Forest Feature Importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": random_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png")
plt.close()

print("\nAll output files generated successfully!")
print("TASK 3 COMPLETED SUCCESSFULLY!")
print("Check the outputs folder.")
