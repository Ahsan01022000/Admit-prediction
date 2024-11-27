import streamlit as st
import joblib
import pandas as pd
model = joblib.load("random.pkl")
st.title("Chance of Admission Predictor")
#user inputs GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research
GRE_Score= st.number_input("Please Enter your GRE Score", min_value=0.0,value=115.0 ,max_value=360.0,step=1.0)
Toefl_Score= st.number_input("Please Enter your Toefl Score", min_value=0.0,value=65.0 ,max_value=120.0,step=1.0)
University_Rating=st.selectbox("Please select your university rating",[1,2,3,4,5])
SOP=st.number_input("please enter your SOP",min_value=0.5,value=3.0,max_value=5.0,step=0.5)
LOR=st.number_input("please enter your LOR",min_value=0.5,value=3.0,max_value=5.0,step=0.5)
Research=st.selectbox("Please select research if done",[0,1])
CGPA=st.number_input("please enter your CPGA",min_value=1.0,value=4.5,max_value=10.0)
if st.button("submit"):
    df=pd.DataFrame([[GRE_Score,Toefl_Score,University_Rating,SOP,LOR,Research,CGPA]])
    result=model.predict(df)[0]
    st.write("The Percentage of getting admitted is",result*100)