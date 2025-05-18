# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Tamil Pavalan M
RegisterNumber:  212223110058
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

**Encoding:**

![image](https://github.com/user-attachments/assets/8d616207-5b17-40f1-807f-fb4e0f7b07fd)

**Head():**

![image](https://github.com/user-attachments/assets/176b6644-01ba-4a58-afa2-bb41dffac21d)

**Info():**

![image](https://github.com/user-attachments/assets/043d1680-3287-42f6-8b5c-af44183eed78)

**isnull().sum():**

![image](https://github.com/user-attachments/assets/e70abdfa-55b8-4b62-bb50-a191ea51b58d)

**Prediction of y:**

![image](https://github.com/user-attachments/assets/43f37d41-9792-4001-99aa-a3e0bfdf7a8c)

**Accuracy:**

![image](https://github.com/user-attachments/assets/a1b94be1-e5ad-4f7d-9745-33848fb2945a)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
