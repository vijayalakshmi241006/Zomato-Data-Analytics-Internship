
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------------
# 1. CREATE OUTPUT FOLDER
# -----------------------------------

output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# -----------------------------------
# 2. LOAD DATASET
# -----------------------------------

df = pd.read_csv("zomato.csv", encoding="latin1")

print("ZOMATO DATA CLEANING AND REPORTING")
print("----------------------------------")

original_rows = len(df)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# -----------------------------------
# 3. CHECK MISSING VALUES
# -----------------------------------

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

missing_before = df.isnull().sum().sum()

# -----------------------------------
# 4. REMOVE DUPLICATES
# -----------------------------------

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

print("\nDuplicates Removed:", duplicates_before)

# -----------------------------------
# 5. HANDLE MISSING VALUES
# -----------------------------------

df["Cuisines"] = df["Cuisines"].fillna("Unknown")

text_columns = [
    "Restaurant Name",
    "City",
    "Address",
    "Locality",
    "Locality Verbose",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu",
    "Rating color",
    "Rating text"
]

for column in text_columns:
    df[column] = df[column].fillna("Unknown")
    df[column] = df[column].astype(str).str.strip()

# -----------------------------------
# 6. CLEAN NUMERICAL VALUES
# -----------------------------------

numeric_columns = [
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# -----------------------------------
# 7. STANDARDIZE TEXT DATA
# -----------------------------------

df["City"] = df["City"].str.title()

df["Restaurant Name"] = (
    df["Restaurant Name"].str.strip()
)

df["Cuisines"] = (
    df["Cuisines"].str.strip()
)

# -----------------------------------
# 8. HANDLE INVALID VALUES
# -----------------------------------

df.loc[
    ~df["Aggregate rating"].between(0, 5),
    "Aggregate rating"
] = df["Aggregate rating"].median()

df.loc[
    ~df["Price range"].between(1, 4),
    "Price range"
] = df["Price range"].median()

df.loc[
    df["Votes"] < 0,
    "Votes"
] = 0

df.loc[
    df["Average Cost for two"] < 0,
    "Average Cost for two"
] = df["Average Cost for two"].median()

# -----------------------------------
# 9. CHECK CLEANED DATA
# -----------------------------------

missing_after = df.isnull().sum().sum()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned rows:", len(df))
print("Cleaned columns:", len(df.columns))

# -----------------------------------
# 10. SAVE CLEANED DATASET
# -----------------------------------

df.to_csv(
    output_folder / "zomato_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

# -----------------------------------
# 11. GENERATE RATING DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    df["Aggregate rating"],
    bins=20,
    kde=True
)

plt.title("Restaurant Rating Distribution")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")

plt.tight_layout()

plt.savefig(
    output_folder / "rating_distribution.png"
)

plt.close()

# -----------------------------------
# 12. GENERATE TOP CITY CHART
# -----------------------------------

top_cities = df["City"].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_cities.values,
    y=top_cities.index
)

plt.title("Top 10 Cities by Restaurant Count")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")

plt.tight_layout()

plt.savefig(
    output_folder / "top_cities.png"
)

plt.close()

# -----------------------------------
# 13. GENERATE CLEANING REPORT
# -----------------------------------

report = f"""
ZOMATO DATA CLEANING REPORT
===========================

Original Records: {original_rows}

Final Records: {len(df)}

Total Columns: {len(df.columns)}

Duplicate Records Removed: {duplicates_before}

Missing Values Before Cleaning: {missing_before}

Missing Values After Cleaning: {missing_after}

Average Restaurant Rating: {df["Aggregate rating"].mean():.2f}

Average Cost for Two: {df["Average Cost for two"].mean():.2f}

Total Cities: {df["City"].nunique()}

Total Restaurants: {df["Restaurant ID"].nunique()}

DATA CLEANING OPERATIONS
------------------------

1. Loaded the Zomato dataset.
2. Checked missing values.
3. Removed duplicate records.
4. Handled missing cuisine values.
5. Standardized text columns.
6. Converted numerical columns.
7. Handled invalid numerical values.
8. Saved the cleaned dataset.
9. Generated visual reports.

STATUS: DATA CLEANING COMPLETED
"""

with open(
    output_folder / "cleaning_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nCleaning Report Generated!")

print("\nAll files saved inside the outputs folder.")

print("\nTASK 4 COMPLETED SUCCESSFULLY!")
