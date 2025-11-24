# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare the Data
Import the required libraries: pandas, numpy, matplotlib, and relevant modules from scikit-learn.
Load the dataset student_scores.csv.
Extract the independent variable (X) and the dependent variable (Y) from the dataset.
2. Split the Data into Training and Testing Sets
Utilize the train_test_split() function to partition the data, allocating two-thirds for training and one-third for testing.
3. Train the Linear Regression Model
Instantiate the LinearRegression() model.
Train the model by fitting it to the training data (X_train and Y_train).
4. Make Predictions and Evaluate the Model
Predict the dependent variable values using X_test.
Calculate the evaluation metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to assess the model’s performance.
5. Visualize the Results
Create a scatter plot to display the training data points.
Overlay the best-fit regression line derived from the trained model.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RANJITH JK
RegisterNumber: 212224230221 
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
<img width="1455" height="971" alt="483861320-46a5c775-e289-47c4-977d-9d77bdd4febd" src="https://github.com/user-attachments/assets/0f6ef12a-6cab-4727-a3a9-daccfb569d48" />

<img width="785" height="573" alt="417874517-5c9a63c0-1115-4a56-9598-d4c616ad40a7" src="https://github.com/user-attachments/assets/2fac11d4-ca2c-479c-a8d5-933d348057a1" />

<img width="791" height="578" alt="417874561-decc794b-aa1f-48a4-9370-593ccf6721d8" src="https://github.com/user-attachments/assets/af64a611-48a1-49a9-ae2d-a2461b0b2990" />


<img width="226" height="35" alt="494710178-cb9dffe7-c7fc-40ed-a397-288fdcc6789e" src="https://github.com/user-attachments/assets/8b495806-338f-40ab-920c-20d8d3f7b321" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
