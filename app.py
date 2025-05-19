import streamlit as st
import pickle
import pandas as pd

with open ('model.pkl','rb')as model_file:
  model=pickle.load(model_file)

st.title('Tip Prediction App')

total_bill=st.number_input('total_bill')
size=st.number_input('size',min_value=1,max_value=10,values=2)
size=st.selectbox('sex',['Male','Female'])
smoker=st.selectbox("Smoker",["Yes","No"])
day=st.selectbox("Day",["Thur","Fri","Sat","Sun"])
time=st.selectbox("TIme",["Lunch","Dinner"])

input_data=pd.DataFrame({
  "total_bill":[total_bill],
  "sex":[sex],
  "smoker":[smoker],
  "day":[day],
  "time":[time],
  "size":[size],
})

sex_maping={'Male':0,'Female':1}
smoker_maping={'NO':0,'Yes':1}
day_maping={'Thur':0,'Fri':1,'Sat':2,'Sun':3}
time_maping={'Lunch':0,'Dinner':1}

input_data['sex']=input_data['sex'].map(sex_maping)
input_data['smoker']=input_data['smoker'].map(smoker_maping)
input_data['day']=input_data['day'].map(day_maping)
input_data['time']=input_data['time'].map(time_maping)

if st.button("predict Tip"):
  prediction=model.predict(input_data)
  st.write(f"Predicted Tip:${round(prediction[0],2)}")
