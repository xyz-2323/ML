# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Display the first few rows of the dataset
print(df.head())

# Display information about the dataset
df.info()

# Check for null values
print(df.isnull().sum())

# Visualizing customer churn based on 'Tenure'
def visualization(x, y, xlabel):
    plt.figure(figsize=(10, 5))
    plt.hist([x, y], color=['red', 'green'], label=['exit', 'not_exit'])
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()
    plt.show()

# Tenure visualization
df_churn_exited = df[df['Exited'] == 1]['Tenure']
df_churn_not_exited = df[df['Exited'] == 0]['Tenure']
visualization(df_churn_exited, df_churn_not_exited, "Tenure")

# Age visualization
df_churn_exited2 = df[df['Exited'] == 1]['Age']
df_churn_not_exited2 = df[df['Exited'] == 0]['Age']
visualization(df_churn_exited2, df_churn_not_exited2, 'Age')

# Preprocessing the data
X = df[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
states = pd.get_dummies(df['Geography'], drop_first=True)
gender = pd.get_dummies(df['Gender'], drop_first=True)

# Combining all features
df = pd.concat([df, gender, states], axis=1)

# Defining the feature set and target variable
X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Male', 'Germany', 'Spain']]
y = df[['Exited']]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the neural network model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Output layer

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
