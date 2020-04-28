#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 


# In[31]:


dataset = pd.read_csv('DATAINDOv2.csv')  
#dataset = pd.read_csv('DATAINDOv3.csv') # memuattanggal
dataset.head()  
dataset.describe()  


# In[32]:


X= dataset.drop('Totalkonfirmed', axis=1) 


# In[33]:


X


# In[34]:


y = dataset['Totalkonfirmed'] 


# In[35]:


y


# In[37]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)  


# In[38]:


# Pada bagian ini kita menggunakan scikit-learn dengan DecisionTreeClassifier
# Untuk mencocokkan model gunakan data latih dengan fungsi fit sbb 

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)  
# Untuk membuat prediksi pada data uji, gunakan metode(fungsi) predict
y_pred = regressor.predict(X_test)  
#Membandingkan data aktual dan hasil model/dugaan
df=pd.DataFrame({'Data aktual':y_test, 'Dugaan':y_pred})  
df    


# In[39]:


from sklearn import metrics  
print('Rata-rata Error absolut:', metrics.mean_absolute_error(y_test, y_pred))  
print('Rata-rata Error kuadrat:', metrics.mean_squared_error(y_test, y_pred))  
print('Akar Rata-rata Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# In[40]:


Indo_kasus = []
total_mati = [] 
lajukematian = []


# In[41]:


def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

Indo_daily_increase = daily_increase(Indo_kasus)


# In[42]:


harihari_sejak_Mar_06 = np.array([i for i in range(len(y))]).reshape(-1, 1)
#Indo_kasus = np.array(Indo_kasus).reshape(-1, 1)
Indo_kasus =y;


# In[44]:


harihari_sejak_Mar_06


# In[45]:


harimendatang = 10
future_forcast = np.array([i for i in range(len(y)+harimendatang)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


# In[49]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(harihari_sejak_Mar_06, Indo_kasus, test_size=0.15, shuffle=False) 


# In[58]:


# svm_confirmed = svm_search.best_estimator_
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=6, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)


# In[59]:


# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))


# In[60]:


start = '3/06/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# In[61]:


# Future predictions using SVM 
print('pendekatan SVM mendatang')
set(zip(future_forcast_dates[-10:], np.round(svm_pred[-10:])))


# In[63]:


# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)


# In[64]:


# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))


# In[65]:


print(linear_model.coef_)


# In[67]:


plt.plot(test_linear_pred)
plt.plot(y_test_confirmed)


# In[68]:


# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)


# In[69]:


bayesian_search.best_params_


# In[70]:


bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))


# In[71]:


plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)


# In[72]:


adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Indo_kasus)
#plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('# hari sejak 3/06/2020', size=30)
plt.ylabel('#Kasus', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[73]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, np.log10(Indo_kasus))
plt.title('Logaritma #kasus terhadap waktu', size=30)
plt.xlabel('Hari sejak 6 Maret 2020', size=30)
plt.ylabel('# Kasus', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[75]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Indo_kasus)
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')
plt.title('#Kasus Coronavirus terhadap waktu', size=30)
plt.xlabel('Hari-hari sejak 3/06/2020', size=30)
plt.ylabel('#Kasus', size=30)
plt.legend(['Kasus Terkonfirmasi', 'Pendekatan regresi polinomial'],prop={'size': 20}) 
plt.yticks(size=20)
plt.show()


# In[76]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Indo_kasus)
plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')
plt.title('# Banyak kasus Coronavirus terhadap waktu', size=30)
plt.xlabel('Waktu',size=30)
plt.ylabel('#Kasus', size=30)
plt.legend(['Kasus Terkonfirmasi', 'Pendekatan polinomial Bayesian Ridge'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[77]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Indo_kasus)
plt.plot(future_forcast, svm_pred, linestyle='dashed', color='purple')
plt.title('# Kasus tiap hari yang terkonfirmasi terhadap waktu', size=30)
plt.xlabel('Hari-hari sejak 3/06/2020', size=30)
plt.ylabel('#Kasus', size=30)
plt.legend(['Kasus terkonfirmasi', 'Pendekatan SVM'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# MEMPELAJARI YANG MENINGGAL

# In[125]:


total_kematian = [] 


# In[126]:


y2 = dataset['Meninggal']


# In[127]:


Indo_kasus =y2;


# In[128]:


y2


# In[129]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(harihari_sejak_Mar_06, Indo_kasus, test_size=0.15, shuffle=False) 


# In[130]:


# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)


# In[131]:


bayesian_search.best_params_


# In[133]:


harimendatang = 10
future_forcast = np.array([i for i in range(len(y)+harimendatang)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


# In[134]:


# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)


# In[135]:


bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))


# In[136]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Indo_kasus)
plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')
plt.title('#Pasien Meninggal terhadap waktu', size=30)
plt.xlabel('Waktu',size=30)
plt.ylabel('#Pasien meninggal', size=30)
plt.legend(['Pasien Meninggal (data)', 'Pendekatan polinomial Bayesian Ridge'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
ax=plt.gca()
ax.set_ylim([0,100])
plt.show()


# MEMEPELAJARI YANG SEMBUH

# In[137]:


y3 = dataset['sembuh']


# In[138]:


Indo_kasus =y3;


# In[139]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(harihari_sejak_Mar_06, Indo_kasus, test_size=0.15, shuffle=False) 


# In[140]:


# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)


# In[141]:


bayesian_search.best_params_


# In[142]:


harimendatang = 10
future_forcast = np.array([i for i in range(len(y3)+harimendatang)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


# In[143]:


# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)


# In[145]:


bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))


# In[147]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Indo_kasus)
plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')
plt.title('#Pasien Sembuh  terhadap waktu', size=30)
plt.xlabel('Waktu',size=30)
plt.ylabel('#Sembuh', size=30)
plt.legend(['# Sembuh (data)', 'Pendekatan polinomial Bayesian Ridge'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
ax=plt.gca()
ax.set_ylim([0,100])
plt.show()


# In[ ]:




