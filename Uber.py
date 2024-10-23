import pandas as pd
import seaborn as sns
from math import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load your data (assuming it's already cleaned)
# data = pd.read_csv('path_to_your_data.csv')  # Replace with your actual data path

# Step 1: Correlation Heatmap
corr = data.corr()
sns.heatmap(data.corr(), annot=True)

# Step 2: Distance Calculation
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    for pos in range(len(longitude1)):
        long1, lati1, long2, lati2 = map(radians, [longitude1[pos], latitude1[pos], longitude2[pos], latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati / 2) ** 2 + cos(lati1) * cos(lati2) * sin(dist_long / 2) ** 2
        c = 2 * asin(sqrt(a)) * 6371  # Distance in kilometers
        travel_dist.append(c)
    return travel_dist

data['dist_travel_km'] = distance_transform(data['pickup_longitude'].to_numpy(),
                                            data['pickup_latitude'].to_numpy(),
                                            data['dropoff_longitude'].to_numpy(),
                                            data['dropoff_latitude'].to_numpy())
# Step 3: Simplified Linear Regression

# Simplified feature set: focusing on the most important ones (distance and hour)
x = data[['dist_travel_km', 'hour']]
y = data['fare_amount']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Applying Linear Regression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Model predictions
pred = regression.predict(X_test)

# Model evaluation
r2 = r2_score(y_test, pred)
MSE = mean_squared_error(y_test, pred)
RMSE = np.sqrt(MSE)

print(f"Linear Regression - R^2 Score: {r2}")
print(f"Linear Regression - Mean Squared Error: {MSE}")
print(f"Linear Regression - Root Mean Squared Error: {RMSE}")

# Step 4: Random Forest Regressor

rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)

# Predict using Random Forest
rf_pred = rf.predict(X_test)

# Random Forest Evaluation
rf_r2 = r2_score(y_test, rf_pred)
rf_MSE = mean_squared_error(y_test, rf_pred)
rf_RMSE = np.sqrt(rf_MSE)

print(f"Random Forest - R^2 Score: {rf_r2}")
print(f"Random Forest - Mean Squared Error: {rf_MSE}")
print(f"Random Forest - Root Mean Squared Error: {rf_RMSE}")
