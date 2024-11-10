import pandas as pd
import seaborn as sns
from math import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load your data (assuming it's already cleaned)
data = pd.read_csv(r"C:\Users\Admin\Documents\ML PRACTICALS\uber.csv")
data

data.head()
data.info()
data.columns

data=data.drop(['Unnamed: 0'], axis=1)
data=data.drop(['key'], axis=1)
data.head()
data.shape
data.dtypes
data.isnull().sum()

data['dropoff_longitude'].fillna(value=data['dropoff_longitude'].mean(), inplace=True)
data['dropoff_latitude'].fillna(value=data['dropoff_latitude'].mean(), inplace=True)


data.pickup_datetime=pd.to_datetime(data.pickup_datetime, errors='coerce')
data= data.assign(hour = data.pickup_datetime.dt.hour,
             day= data.pickup_datetime.dt.day,
             month = data.pickup_datetime.dt.month,
             year = data.pickup_datetime.dt.year,
             dayofweek = data.pickup_datetime.dt.dayofweek)
data=data.drop(['pickup_datetime'], axis=1)
data.head()



#2 identify outliers
data.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20)) 

#Using the InterQuartile Range to fill the values
def remove_outlier(data1 , col):
    Q1 = data1[col].quantile(0.25)
    Q3 = data1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    data1[col] = np.clip(data1[col] , lower_whisker , upper_whisker)
    return data1

def treat_outliers_all(data1, col_list):
    for col in col_list:
        data1 = remove_outlier(data1, col)
    return data1

# Applying outlier treatment to all columns in the dataset
data = treat_outliers_all(data, data.columns)
data.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20))

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
#linear regression
x = data.drop('fare_amount', axis=1)
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
