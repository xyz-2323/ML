import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
print(data.head())

# Display correlation matrix
correlation_matrix = data.corr()['Outcome']
print(correlation_matrix)

# Plot heatmap of correlations
sns.heatmap(data.corr(), annot=True)
plt.show()

# Prepare the data for training
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with a scaler and KNN classifier
scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(operations)

# Set up GridSearchCV for hyperparameter tuning
k_values = list(range(1, 20))
param_grid = {'knn__n_neighbors': k_values}
full_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

# Fit the model
full_classifier.fit(X_train, y_train)

# Display the best estimator parameters
best_params = full_classifier.best_estimator_.get_params()
print(best_params)

# Make predictions on the test set
y_pred = full_classifier.predict(X_test)

# Calculate accuracy and error rate
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Error Rate: {error_rate}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
CMD = ConfusionMatrixDisplay(cm).plot()

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Display classification report
print(classification_report(y_test, y_pred))
