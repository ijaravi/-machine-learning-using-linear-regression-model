import pandas as pd 
import matplotlib.pyplot as plt 

data=pd.read_csv('advertising.csv')
data.head()

data.shape
fig,axs=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0], figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

feature_cols=['TV']
X=data[feature_cols]
y=data.Sales
from sklearn.linear_model import LinearRegression
lm=LinearRegression()#y=f(x)
lm.fit(X,y)# fit means wwe are training the data

print(lm.intercept_)# b is thee intercept 
print(lm.coef_)# a is the coefficent 

6.97482+0.55464*50
x_new=pd.DataFrame({'TV':[50]})
x_new.head()

lm.predict(x_new)

x_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()

preds=lm.predict(x_new)
preds

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth=2)

import statsmodels.formula.api as smf 
lm=smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int()

lm.pvalues

lm.rsquared

feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)

lm=smf.ols(formula='Sales ~ TV + Radio + Newspaper',data=data).fit()
lm.conf_int()
lm.summary

lm=smf.ols(formula='Sales ~ TV+Radio',data=data).fit()
lm.rsquared

import numpy as np 

np.random.seed(12345)# set a seed reproducility

nums =np.random.rand(len(data))#create a series of booleanin roughly half to be large 
mask_large=nums > 0.5

#intially to set size to small, then change roughly to the half to largee 
data['Size']='small'
data.loc[mask_large,'Size']='Large'
data.head()

data['IsLarge']=data.Size.map({'small':0,'Large':1})    
data.head()

#create a new seriees 
feature_cols=['TV','Radio','Newspaper','IsLarge']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)

print(feature_cols,lm.coef_)

np.random.seed(123456)

nums=np.random.rand(len(data))
mask_suburban=(nums>0.33) & (nums<0.66) 
mask_urban=nums>0.66 
data['Area']='rural'
data.loc[mask_suburban,'Area']='suburban'
data.loc[mask_urban,'Area']='urban'
data.head()

area_dummies=pd.get_dummies(data.Area,prefix='Area').iloc[:,1:]
area_dummies
#concate the dummy variable colums into the original data frame
data= pd.concat([data,area_dummies],axis=1)
data.head()
#create x and y 
feature_cols=['TV','Radio','Newspaper','IsLarge','Area_suburban','Area_urban']
X=data[feature_cols]
y=data.Sales
 #instantiatw,fit 
lm=LinearRegression()
lm.fit(X,y)# by this we are training the model
#print coeffients 
print(feature_cols,lm.coef_)