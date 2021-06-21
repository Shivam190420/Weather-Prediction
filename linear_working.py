import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder


#--------------------data read for temperature---------------------------
data=pd.read_csv("weatherHistory.csv")
data=data[list(data.dtypes[data.dtypes!='object'].index)]
temp_y = data.pop('Temperature (C)')
temp_x = data
data.columns
#------------------------data read for weather------------------------------
data=pd.read_csv("weatherHistory.csv")
weth_y = data.pop('Summary')
data=data[list(data.dtypes[data.dtypes!='object'].index)]
weth_x = data.drop(['Apparent Temperature (C)'], axis=1)
le=LabelEncoder()
weth_y=le.fit_transform(weth_y) 

#--------------------------------TEMP predict(Linear)-------------------------------
x_train,x_test,y_train,y_test=train_test_split(temp_x,temp_y,test_size=0.2)

l1=LinearRegression()
l1.fit(x_train,y_train)
print("Score for temperture using LinearRegeression= ",l1.score(x_train,y_train))
p=l1.predict(x_test)
np.mean((p-y_test)**2)


temp_x.columns
v=[7.89,0.89,14.1197,251,15.8263,0,1015.13]    #for prediction as per values in x
v=np.array(v)
v=v.reshape(1,len(v))
pred=l1.predict(v)
print(pred[0],"Celcious")


#--------------------------------weather predict(Linear)-------------------------------
x_train,x_test,y_train,y_test=train_test_split(weth_x,weth_y,test_size=0.2)

l2=LinearRegression()
l2.fit(x_train,y_train)
print("Score for weather using LinearRegeression= ",l2.score(x_train,y_train))
p=l2.predict(x_test)
np.mean((p-y_test)**2)

weth_x.columns
v=[9.47222,0.89,14.1197,251,15.8263,0,1015.13]    #for prediction as per values in x
v=np.array(v)
v=v.reshape(1,len(v))
predict_value=int(l2.predict(v))
predict_value=np.array(predict_value)
predict_value=predict_value.reshape(1,1)


predict_value=le.inverse_transform(predict_value) 
print(predict_value[0])


#-------------------------------temp predict(svr)--------------------------
svr_y=np.array(temp_y)
svr_x=temp_x
svr_y=svr_y.reshape(len(svr_y),1)
sxs=StandardScaler()  #WE HAVE CREATED TWO SXS AND SYS AS EACH IS GOING TO FITTED TO CERTAIN MATRIX
sys=StandardScaler()
svr_x=sxs.fit_transform(svr_x)
svr_y=sys.fit_transform(svr_y)

x_train,x_test,y_train,y_test=train_test_split(svr_x,svr_y,test_size=0.2)
#REGRESSION
from sklearn.svm import SVR
svr1=SVR(kernel='rbf')
svr1.fit(x_train,y_train)
print("Score for temperture using SVR= ",svr1.score(x_train,y_train))
p=svr1.predict(x_test)
np.mean((p-y_test)**2)


v=[7.89,0.89,14.1197,251,15.8263,0,1015.13]    #for prediction as per values in x
v=np.array(v)
v=v.reshape(1,len(v))
predict_svr=svr1.predict(sxs.transform(v))
predict_svr=sys.inverse_transform(predict_svr)
print(predict_svr[0],"Celcious")


#------------------------------weather predict(svr)-----------------------
svr_y1=np.array(weth_y)
svr_x1=weth_x
svr_y1=svr_y1.reshape(len(svr_y1),1)
svr_x1=sxs.fit_transform(svr_x1)
svr_y1=sys.fit_transform(svr_y1)

x_train,x_test,y_train,y_test=train_test_split(svr_x1,svr_y1,test_size=0.2)
from sklearn.svm import SVR
svr2=SVR(kernel='rbf')
svr2.fit(x_train,y_train)
print("Score for Weather using SVR= ",svr2.score(x_train,y_train))
p=svr1.predict(x_test)
np.mean((p-y_test)**2)


v=[9.47222,0.89,14.1197,251,15.8263,0,1015.13]    #for prediction as per values in x
v=np.array(v)
v=v.reshape(1,len(v))
predict_svrwe=int(svr2.predict(sxs.transform(v)))

predict_svrwe=np.array(predict_svrwe)
predict_svrwe=predict_svrwe.reshape(1,1)
predict_svrwe=le.inverse_transform(predict_svrwe) 
print(predict_svrwe[0])


#-----------------------------------END--------------------------------
plt.scatter(data[''],0,color='red')
plt.plot(val_y,0,color='blue')
plt.xlabel('Predict')
plt.ylabel('original')
plt.title("title here")
plt.show()

