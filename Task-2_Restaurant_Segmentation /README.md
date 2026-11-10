# Task 2 – Restaurant Segmentation Using K-Means

## Project Overview

This task focuses on grouping Zomato restaurants based on their pricing, ratings, customer votes and price range using the K-Means clustering algorithm.

The objective is to identify groups of restaurants with similar characteristics through unsupervised machine learning.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset

The cleaned Zomato restaurant dataset containing restaurant ratings, average cost for two, votes and price range was used.

## Methodology

1. Loaded the cleaned Zomato dataset.
2. Selected relevant restaurant features.
3. Handled invalid and missing records for clustering.
4. Calculated cost percentiles within each currency.
5. Applied feature scaling using StandardScaler.
6. Used the Elbow Method to examine the number of clusters.
7. Applied K-Means clustering with three clusters.
8. Generated cluster visualisations and exported the results.

## Results

A total of 7,385 restaurants were selected and grouped into three clusters.

- Cluster 0: 2,285 restaurants
- Cluster 1: 2,754 restaurants
- Cluster 2: 2,346 restaurants

## Generated Outputs

- `restaurant_clusters.csv` – Dataset containing cluster assignments.
- `elbow_method.png` – Elbow method visualisation.
- `restaurant_segmentation.png` – Restaurant cluster visualisation.

## Conclusion

K-Means clustering helped identify groups of restaurants with similar pricing, rating and customer engagement characteristics. The results provide a better understanding of restaurant segments within the Zomato dataset.

## Status

Completed
