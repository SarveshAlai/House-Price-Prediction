# House-Price-Prediction
A Machine Learning model to predict house prices using Kaggle dataset
A machine learning project to predict house prices based on various features using regression models.

📌 Project Overview
This project aims to predict house prices using different machine learning techniques. The dataset contains various attributes of houses, such as lot size, number of rooms, year built, and more. The model is trained and evaluated to ensure accurate price predictions.
🚀 Features of the Project
✅ Data Cleaning & Preprocessing
✅ Exploratory Data Analysis (EDA)
✅ Feature Engineering
✅ Model Training & Evaluation
✅ Performance Metrics Calculation
The dataset used in this project is a CSV file containing house attributes.
It includes features like LotArea, YearBuilt, TotalRooms, GarageArea, etc.
The dataset is loaded using:
import pandas as pd
df = pd.read_csv("/content/train.csv")  # Adjust path if needed
df.head()
⚙ Machine Learning Models Used
The following regression models were implemented:
1️⃣ Linear Regression
2️⃣ Random Forest Regressor
3️⃣ Gradient Boosting Regressor
📊 Performance Metrics
To evaluate model accuracy, we used:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
📌 Interpretation of Metrics:

MAE (Mean Absolute Error): Lower is better.
MSE (Mean Squared Error): Lower is better.
R² Score: Closer to 1 means a better model.
🛠 Technologies Used
🔹 Python 🐍
🔹 Pandas & NumPy
🔹 Matplotlib & Seaborn
🔹 Scikit-learn
🔹 Google Colab
📜 Author
👤 Sarvesh Alai
