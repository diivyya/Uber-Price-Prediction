import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv('taxi.csv') #to load data 
print(data.head()) #to read data and print as well

data_x = data.iloc[:,0:-1].values #to take features in data_x iloc will take :(all rows) & all columns except last(0:-1) their values
data_y = data.iloc[:,-1].values #the column we want to predict
print(data_y)

#by sklearn we can now split our data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0) #this will take x & y & we said 30% of data will be testing data & random state will help to check predicted data is how much correct

reg = LinearRegression()
reg.fit(X_train,Y_train) #model is trained

#checks performance of model(accuracy)
print("Train Score: ",reg.score(X_train,Y_train))
print("Test Score: ",reg.score(X_test,Y_test))

#pickle module is used to keep data at a place to use at flask
pickle.dump(reg, open('taxi.pkl','wb')) #creates file and keeps data there

#now testing
model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1770000,6000,85]])) #finds preidction on the basis of values


