
# TASK 2: RESTAURANT SEGMENTATION USING K-MEANS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("ZOMATO RESTAURANT SEGMENTATION")
print("--------------------------------")

# 1. Load the cleaned dataset
df = pd.read_csv("zomato_cleaned.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Total restaurants:", len(df))

# 2. Create output folder
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# 3. Select required columns
required_columns = [
    "Average Cost for two",
    "Aggregate rating",
    "Votes",
    "Price range"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column not found: {column}")

# 4. Convert selected columns into numeric format
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 5. Remove invalid records for clustering
data = df.dropna(subset=required_columns).copy()

# Keep only restaurants with valid ratings and costs
data = data[
    (data["Aggregate rating"] > 0) &
    (data["Aggregate rating"] <= 5) &
    (data["Average Cost for two"] > 0) &
    (data["Votes"] >= 0) &
    (data["Price range"] >= 1) &
    (data["Price range"] <= 4)
].copy()

if len(data) < 3:
    raise ValueError("Not enough valid restaurant records for clustering.")

print("Restaurants selected for clustering:", len(data))

# 6. Calculate cost percentile within each currency
# This avoids directly comparing different currencies.
if "Currency" in data.columns:
    currency_group = data["Currency"].fillna("Unknown")
    data["Cost Percentile"] = (
        data.groupby(currency_group)["Average Cost for two"]
        .rank(pct=True)
    )
else:
    data["Cost Percentile"] = (
        data["Average Cost for two"].rank(pct=True)
    )

# 7. Prepare features for K-Means
# Log transformation reduces the effect of very high vote counts.
features = data[
    [
        "Cost Percentile",
        "Aggregate rating",
        "Votes",
        "Price range"
    ]
].copy()

features["Votes"] = np.log1p(features["Votes"])

# 8. Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Feature scaling completed!")

# 9. Elbow Method
inertia_values = []

max_clusters = min(8, len(data) - 1)

for k in range(1, max_clusters + 1):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(scaled_features)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, max_clusters + 1),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, max_clusters + 1))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "elbow_method.png")
)
plt.close()

print("Elbow method graph generated!")

# 10. Apply K-Means with 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(scaled_features)

# 11. Save segmented dataset
data.to_csv(
    os.path.join(output_folder, "restaurant_clusters.csv"),
    index=False
)

print("Restaurant cluster dataset saved!")

# 12. Generate cluster visualisation
plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    data["Cost Percentile"],
    data["Aggregate rating"],
    c=data["Cluster"],
    cmap="viridis",
    s=35,
    alpha=0.7
)

plt.title("Zomato Restaurant Segmentation")
plt.xlabel("Cost Percentile within Currency")
plt.ylabel("Aggregate Rating")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "restaurant_segmentation.png")
)
plt.close()

print("Restaurant segmentation graph generated!")

# 13. Display cluster summary
cluster_summary = data.groupby("Cluster").agg(
    Restaurant_Count=("Restaurant ID", "count")
    if "Restaurant ID" in data.columns
    else ("Aggregate rating", "count"),
    Average_Rating=("Aggregate rating", "mean"),
    Average_Cost_Percentile=("Cost Percentile", "mean"),
    Average_Votes=("Votes", "mean"),
    Average_Price_Range=("Price range", "mean")
).round(2)

print("\nCLUSTER SUMMARY")
print(cluster_summary)

# 14. Finish
print("\nTASK 2 COMPLETED SUCCESSFULLY!")
print("All output files are saved inside the outputs folder.")
