import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report

# Ignore warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('emails.csv')

# Data preprocessing
df.drop(['Email No.'], axis=1, inplace=True)
X = df.drop(['Prediction'], axis=1)
y = df['Prediction']

# Scale features
X = scale(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# KNN evaluation
print("KNN accuracy = ", metrics.accuracy_score(y_test, y_pred_knn))
print("KNN Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_knn))

# SVM model (default)
model_svc = SVC(C=1)
model_svc.fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_test)

# SVM evaluation
print("SVM accuracy = ", metrics.accuracy_score(y_test, y_pred_svc))
print("SVM Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_svc))

# SVM with different kernels
kernels = ['poly', 'rbf', 'sigmoid']
for kernel in kernels:
    start = time.time()
    model = SVC(kernel=kernel, C=2)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    end = time.time()
    print(f"SVM with {kernel} kernel accuracy = {round(acc * 100, 1)}% in {end - start:.5f} sec")

# SVM with LinearSVC
start = time.time()
model_linear = LinearSVC(C=2)
model_linear.fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)
acc_linear = accuracy_score(y_test, pred_linear)
end = time.time()
print(f"LinearSVC accuracy = {round(acc_linear * 100, 1)}% in {end - start:.5f} sec")

# Feature scaling with different scalers
scalers = [MinMaxScaler(), RobustScaler(), StandardScaler()]
scaler_names = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']

for scaler, name in zip(scalers, scaler_names):
    x = df.drop('Prediction', axis=1)
    y = df['Prediction']
    x = scaler.fit_transform(x)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    
    print(f"Using {name}")
    
    for kernel in kernels:
        start = time.time()
        model = SVC(kernel=kernel, C=3)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        end = time.time()
        print(f"SVM with {kernel} kernel accuracy = {round(acc * 100, 1)}% in {end - start:.5f} sec")
    
    start = time.time()
    model_linear = LinearSVC(C=3)
    model_linear.fit(X_train, y_train)
    pred_linear = model_linear.predict(X_test)
    acc_linear = accuracy_score(y_test, pred_linear)
    end = time.time()
    print(f"LinearSVC accuracy = {round(acc_linear * 100, 1)}% in {end - start:.5f} sec")

# Final classification report
y_pred2 = model_linear.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred2))
