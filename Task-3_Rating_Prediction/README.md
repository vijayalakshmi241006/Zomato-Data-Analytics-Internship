# Task 3 – Restaurant Rating Prediction Using Machine Learning

## Project Overview
This task focuses on predicting restaurant ratings using machine learning techniques. Two regression algorithms, Linear Regression and Random Forest Regression, are implemented and compared using the Zomato restaurant dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset
The cleaned Zomato restaurant dataset was used. It contains information about restaurant pricing, customer votes, price range, service availability and aggregate ratings.

## Features Used
- Average Cost for two
- Votes
- Price range
- Has Table booking
- Has Online delivery

**Target Variable:** Aggregate rating

## Methodology
1. Loaded the cleaned Zomato dataset.
2. Selected relevant features for prediction.
3. Excluded unrated restaurants.
4. Handled missing values.
5. Split the dataset into training and testing sets.
6. Applied feature scaling for Linear Regression.
7. Trained Linear Regression and Random Forest models.
8. Evaluated and compared model performance using MAE, RMSE and R² Score.

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.372 | 0.470 | 0.285 |
| Random Forest | 0.300 | 0.406 | 0.467 |

Random Forest performed better than Linear Regression on the test dataset.

## Generated Outputs
- `model_comparison.csv` – Comparison of model performance.
- `actual_vs_predicted.png` – Visual comparison of actual and predicted ratings.
- `feature_importance.png` – Importance of features in Random Forest prediction.

## Conclusion
The task demonstrates how machine learning regression algorithms can be used to estimate restaurant ratings based on selected restaurant characteristics. Among the two models tested, Random Forest achieved better prediction performance.

## Status
Completed
