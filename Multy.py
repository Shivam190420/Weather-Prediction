  #importing library and DataSet
# Three IMP. Librarys
""" :: DATA PREPROCESSING PROGRAM :: """
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd                  #use to import dataset and manage them

# to access dataset we need to set a working directory first

data=pd.read_csv("weatherHistory.csv")
# First drop the maxtempm and mintempm from the dataframe
x = data.drop(['Precip Type', 'Daily Summary','Formatted Date'], axis=1)
y = data['Precip Type']

from sklearn.preprocessing import LabelEncoder,OneHotEncoder  # use to incode values
le=LabelEncoder()
x.iloc[:,0]=le.fit_transform(x.iloc[:,0])
y=le.fit_transform(y.astype(str)) 

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
[('oh', OneHotEncoder(), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
x = ct.fit_transform(x)
x=x.todense()
# OVERCOME DUMMY VARIABLE TRAP
x=x[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#FEATURE SCALLING
'''from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test) #NO NEED TO FIT IT AS ALREADY FITED TO TRAINING SET

no need to fiture scale y as it is already a kind of fitted'''
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))

predict_value=lr.predict(x_test)
predict_value=le.inverse_transform(predict_value) 

#USING BACKWARD ELIMINATION
import statsmodels.api as sm
x=np.append(np.ones((96453,1)).astype(int),values=x,axis=1)  #explination in copy
'''NOTE THAT WE HAVE PUT NP.ONES IN FRONT AS WE WANT TO ADD ARRAY OF 1 IN FRONT OF X
TO ADD IT BACK WRITE IT AS:--x=np.append(x , values=np.ones((50,1)).astype(int) , axis=1) '''

x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,34,36]]

'''we need to select a significant level so we take it as 5%[0.05]'''
regressor=sm.OLS(endog=y,exog=x_opt).fit() # Step 2
k=regressor.summary()
print(k)

