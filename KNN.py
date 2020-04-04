import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("D:\social.csv")
df[0:6]

x = df.iloc[:,[2,3]].values
x[0:6]

y = df['Purchased'].values
y

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state= 0)

len(x_train)

len(x_test)

ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

x_test[:6]

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

knn.fit(x_train,y_train)

knn.predict(x_test)

knn.score(x_test,y_pred)

y_test

cm = confusion_matrix(y_test,knn.predict(x_test))
cm

import seaborn as sn


plt.title(" Confusion Matrix")
sn.heatmap(cm,annot=True)
plt.xlabel('Prediction')
plt.ylabel("Truth")



