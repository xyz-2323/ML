import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\Admin\Documents\ML PRACTICALS\sales_data_sample.csv",encoding='latin1')

df.head

df.isnull().sum()

df.dtypes

df_drop = [
    'ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 
    'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 
    'ORDERNUMBER'
]
df = df.drop(df_drop, axis=1)


df.head()


df.dtypes

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
df['YEAR'] = df['ORDERDATE'].dt.year
df['MONTH'] = df['ORDERDATE'].dt.month
df['DAY'] = df['ORDERDATE'].dt.day
df.head()

df=df.drop(["ORDERDATE"],axis=1)


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# List of columns to encode
columns_to_encode = ['STATUS', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE']

# Apply label encoding to each specified column
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Display the first few rows to verify encoding
print(df.head())


df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes
df.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(df)



from sklearn.cluster import KMeans
wcss = []  # Initialize an empty list to store WCSS values
for i in range(1, 11):  # Loop over a range of cluster numbers from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # Initialize KMeans with i clusters
    kmeans.fit(df)  # Fit KMeans to the scaled data
    wcss.append(kmeans.inertia_)  # Append the WCSS (inertia) to the list




import matplotlib.pyplot as plt  # Correct import for plotting

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


optimal_k = 5  # Replace with the chosen number of clusters based on the inertia plot
kmeans = KMeans(n_clusters=optimal_k, n_init=10)
kmeans.fit(df)




import pandas as pd

# If df is a NumPy array, convert it to a DataFrame
df = pd.DataFrame(df)

# Now reset the index and assign the cluster labels
df = df.reset_index(drop=True)

# Assign the cluster labels to the 'Cluster' column
df['Cluster'] = kmeans.labels_

# Check the result
print(df.head())















