import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\\Users\\Admin\\Documents\\ML PRACTICALS\\diabetes.csv")

df.head

x=df.drop('Outcome',axis=1)
y=df['Outcome']

import seaborn as sns
sns.countplot(x=y)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

x_sacled=scaler.fit_transform(x)

# cross validation
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,classification_report, confusion_matrix

y_pred=knn.predict(x_test)

confusion_matrix(y_test,y_pred)

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)

import numpy as np
import matplotlib.pyplot as plt

error=[]
for k in range(1,41):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred_k=knn.predict(x_test)
    error.append(np.mean(y_pred_k != y_test))

error

knn=KNeighborsClassifier(n_neighbors=22)

knn.fit(x_train,y_train)
pred=knn.predict(x_test)

acc=accuracy_score(y_test,y_pred)
acc


print(classification_report(y_test,y_pred))



error_rate=1-acc

error_rate







