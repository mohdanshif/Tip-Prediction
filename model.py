import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

data=sns.load_dataset('tips')

sex_maping={'male':0,'female':1}
smoker_maping={'No':0,'Yes':1}
day_maping={'Thur':0,'Fri':1,'sat':2,'Sun':3}
time_maping={'Lunch':0,'Dinner':1}

data['sex']=data['sex'].map(sex_maping)
data['smoker']=data['smoker'].map(smoker_maping)
data['day']=data['day'].map(day_maping)
data['time']=data['time'].map(time_maping)

x=data.drop(columns=['tip'])
y=data['tip']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestRegressor()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
print(f"Mean Absolute Error:{mae}")

with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)
