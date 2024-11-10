import pandas as pd
import numpy as np

df = pd.read_csv("emails.csv")

df.head()
df.shape

x = df.drop(['Email No.', 'Prediction'], axis=1)
y = df['Prediction']

x.dtypes

import seaborn as sns
sns.countplot(x=y)

from sklearn.preprocessing import scale
x = scale(x)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("prediction", y_pred)

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

print("KNN accuracy:", accuracy_score(y_test, y_pred))

print("Confusion matrix", metrics.confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

accuracy_score(y_test, y_pred)

import numpy as np
import matplotlib.pyplot as plt

if isinstance(X_test, pd.Series):
    X_test = X_test.to_frame().T
elif len(X_test.shape) == 1:
    X_test = X_test.values.reshape(-1, 1)

# Check shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Error array to store error rates for each value of k
error = []

# Iterate over a range of k values
for k in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Predict and calculate the error
    pred = knn.predict(X_test)
    error.append(np.mean(pred != y_test))

error

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy_score(y_test, y_pred)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_score(y_test, y_pred)

# svm = SVC(kernel='rbf')
# svm = SVC(kernel='poly') we can take this also as a kernel and check accuracy
