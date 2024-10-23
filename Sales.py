import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape', parse_dates=['ORDERDATE'])

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop unnecessary columns
df_drop = [
    'ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 
    'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 
    'ORDERNUMBER'
]
df = df.drop(df_drop, axis=1)

# Check the shape of the DataFrame
print(df.shape)

# Check for any remaining missing values
print(df.isna().sum())

# Function to create bar plots
def barplot_visualization(x):
    fig = plt.Figure(figsize=(12, 6))
    fig = px.bar(x=df[x].value_counts().index, y=df[x].value_counts(), color=df[x].value_counts().index, height=600)
    fig.show()

# Visualize COUNTRY and STATUS
barplot_visualization('COUNTRY')
barplot_visualization('STATUS')

# Drop unbalanced feature
df.drop(columns=['STATUS'], axis=1, inplace=True)
print('Columns resume:', df.shape[1])

# Visualize PRODUCTLINE and DEALSIZE
barplot_visualization('PRODUCTLINE')
barplot_visualization('DEALSIZE')

# Prepare data by creating dummy variables
def dummies(x):
    dummy = pd.get_dummies(df[x])
    df.drop(columns=x, inplace=True)
    return pd.concat([df, dummy], axis=1)

df = dummies('COUNTRY')
df = dummies('PRODUCTLINE')
df = dummies('DEALSIZE')

# Display the updated DataFrame
print(df.head())

# Convert PRODUCTCODE to categorical codes
df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes

# Drop unnecessary date-related columns
df.drop('ORDERDATE', axis=1, inplace=True)
df.drop('QTR_ID', axis=1, inplace=True)

# Check the shape after dropping columns
print(df.shape)

# Use K-Means algorithm for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Determine the optimal number of clusters using inertia
scores = []
range_values = range(1, 15)
for i in range_values:
    kmeans = KMeans(n_clusters=i, n_init=10)  # Set n_init to suppress warning
    kmeans.fit(df_scaled)
    scores.append(kmeans.inertia_)

# Plot the inertia scores
plt.figure(figsize=(12, 6))
plt.plot(range_values, scores, marker='o')
plt.title('K-Means Clustering: Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range_values)
plt.grid()
plt.show()

# Optional: Fit the KMeans model with the optimal number of clusters (choose based on the plot)
optimal_k = 5  # Replace with the chosen number of clusters based on the inertia plot
kmeans = KMeans(n_clusters=optimal_k, n_init=10)
kmeans.fit(df_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Display the first few rows of the DataFrame with cluster labels
print(df.head())
