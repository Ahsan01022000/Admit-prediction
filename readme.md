# Implementation of Random Forest Algorithm using Python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

%%capture
pip install pandas numpy matplotlib seaborn

df=pd.read_csv("graduate_admission.csv")

# drop unnecessary columns
df.drop(columns=["Unnamed: 0","Serial No."], inplace=True)

# Encoding- we don't have cat to encode
# df.rename()
# Feature Scaling not required since we are using ensemble model
X= df.drop(columns="Chance of Admit ")
y= df["Chance of Admit "]

# We need to split the data into train test
from sklearn.model_selection import train_test_split

%%capture
pip install scikit-learn

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Model Training
from sklearn.ensemble import RandomForestRegressor

# create an object of the RandomForest Regressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

# Testing
y_pred=rf.predict(X_test)

# Evaluate the performance of Regressor
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print("mean squared error",mse)
rmse = np.sqrt(mse)
print("Root Mean Squared error is",rmse)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


y_pred1 = lr.predict(X_test)

# Evaluate the performance of Regressor
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred1)
print("mean squared error",mse)
rmse = np.sqrt(mse)
print("Root Mean Squared error is",rmse)


from sklearn.tree import DecisionTreeRegressor
dtf=DecisionTreeRegressor()
dtf.fit(X_train,y_train)


y_pred2 = dtf.predict(X_test)

# mean, std for performing standardization # fit
# transform will do the transformation
# Evaluate the performance of Regressor
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred2)
print("mean squared error",mse)
rmse = np.sqrt(mse)
print("Root Mean Squared error is",rmse)


# build a streamlit application with random forest model
import joblib
joblib.dump(rf, "random.pkl")


-- streamlit code

import streamlit as st
import joblib
import pandas as pd


model = joblib.load("randomforest_model.pkl")
st.title("Chance of Admission Predictor")
# streamlit run app.py
# GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research
GRE_Score = st.number_input("Please enter your GRE Score", min_value=0.0, max_value = 360.0,step=1.0)
Toefl_Score=st.number_input("Please enter your Toefl Score", min_value=0.0, max_value = 120.0,step=1.0)
University_rating = st.selectbox("Please select a rating",[1,2,3,4,5])
SOP=st.number_input("enter your SOP value",min_value=0.0, max_value=5.0, step=0.5)
LOR=st.number_input("enter your LOR value",min_value=0.0, max_value=5.0, step=0.5)
CGPA = st.number_input("enter your CGPA score",min_value=1.0, max_value=10.0)
Research = st.selectbox("Select a research value",[0,1])

if st.button("submit"):
    df = pd.DataFrame([[GRE_Score,Toefl_Score,University_rating,SOP,LOR,CGPA,Research]])
    result=model.predict(df)[0] # [0.93]
    st.write("The chance of getting admitted would be", result)











