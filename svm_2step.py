import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
train_data = pd.read_csv('train_upd.csv')
y = pd.read_csv('train_upd.csv')

y = y["Congestion_Type"]
y[y == "4G_BACKHAUL_CONGESTION"] = "C"
y[y == "4G_RAN_CONGESTION"] = "C"
y[y == "3G_BACKHAUL_CONGESTION"] = "C"
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(train_data['ran_vendor'].astype(str))
train_data["ran_vendor"] = mvar47

le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(y.astype(str))
y = mvar47
# print(le.inverse_transform(y))
train_data['total_bytes'] = 0
byte_cols = ['web_browsing_total_bytes',
 'video_total_bytes',
 'social_ntwrking_bytes',
 'cloud_computing_total_bytes',
 'web_security_total_bytes',
 'gaming_total_bytes',
 'health_total_bytes',
 'communication_total_bytes',
 'file_sharing_total_bytes',
 'remote_access_total_bytes',
 'photo_sharing_total_bytes',
 'software_dwnld_total_bytes',
 'marketplace_total_bytes',
 'storage_services_total_bytes',
 'audio_total_bytes',
 'location_services_total_bytes',
 'presence_total_bytes',
 'advertisement_total_bytes',
 'system_total_bytes',
 'voip_total_bytes',
 'speedtest_total_bytes',
 'email_total_bytes',
 'weather_total_bytes',
 'media_total_bytes',
 'mms_total_bytes',
 'others_total_bytes']
train_data['total_bytes_mod'] = 0
for i in byte_cols:
    train_data['total_bytes'] += train_data[i]
train_data.total_bytes_mod = train_data.total_bytes/1024 #for GB/s
train_data.total_bytes_mod = (train_data.total_bytes_mod/(train_data.par_min*60))

import datetime
temp =[]
for i in range(len(train_data)):
   temp.append(datetime.datetime(train_data['par_year'][i],train_data['par_month'][i],train_data['par_day'][i]).weekday())
train_data["week_day"] = temp
x=train_data.drop(['Congestion_Type','cell_name','par_year','par_month','par_day'],axis=1)
# In[39]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=20)


# In[81]:


from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
regressor=RandomForestClassifier(n_estimators=700,max_features=3,max_depth=26,random_state=0,n_jobs=-1)
xgb1 = xgb.XGBClassifier(booster='gbtree', colsample_bylevel=1,
    colsample_bytree=1, gamma=0, learning_rate=0.15,
    max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
    n_estimators=700, n_jobs=3, nthread=1, objective='binary:logistic',
    random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
    seed=None, subsample=1)
#my_scorer = make_scorer(matthews_corrcoef)
#print(cross_val_score(xgb1, x, y,scoring= my_scorer, cv=4, verbose=10))


# In[82]:


#xgb1.fit(x,y)
regressor.fit(x_train,y_train)

# In[209]:


y=le.inverse_transform(y)


# In[210]:


y


# In[211]:


#pred = xgb1.predict(x)
pred = regressor.predict(x_test)
print(matthews_corrcoef(pred,y_test))


# In[223]:


train_data.head()
new_data = train_data[train_data['Congestion_Type']!= "NC"]


# In[224]:


y2 = new_data['Congestion_Type']


# In[225]:


le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(y2.astype(str))
y2 = mvar47
# le.inverse_transform(y2)


# In[226]:


# y2[1:10]


# In[227]:


x2=new_data.drop(['Congestion_Type','cell_name','par_year','par_month','par_day'],axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[102]:
x_train ,x_test,y_train,y_test=train_test_split(x2,y2,test_size=0.20,random_state=20)
regressor2 = RandomForestClassifier(n_estimators=700,max_features=3,max_depth=26,random_state=0,n_jobs=-1)
xgb2 = xgb.XGBClassifier(booster='gbtree', colsample_bylevel=1,
    colsample_bytree=1, gamma=0, learning_rate=0.15,
    max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
    n_estimators=700, n_jobs=3, nthread=1, objective='multi:softmax',
    random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
    seed=None, subsample=1)
#xgb2.fit(x2,y2)
regressor2.fit(x2,y2)

# In[228]:

pred2 = regressor2.predict(x2)
#pred2 = xgb2.predict(x2)
len(pred2[pred2 == 2])


# In[229]:


test = pd.read_csv('train_upd.csv')


# In[230]:


le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(test['ran_vendor'].astype(str))
test["ran_vendor"] = mvar47


# In[231]:


test['total_bytes'] = 0
byte_cols = ['web_browsing_total_bytes',
 'video_total_bytes',
 'social_ntwrking_bytes',
 'cloud_computing_total_bytes',
 'web_security_total_bytes',
 'gaming_total_bytes',
 'health_total_bytes',
 'communication_total_bytes',
 'file_sharing_total_bytes',
 'remote_access_total_bytes',
 'photo_sharing_total_bytes',
 'software_dwnld_total_bytes',
 'marketplace_total_bytes',
 'storage_services_total_bytes',
 'audio_total_bytes',
 'location_services_total_bytes',
 'presence_total_bytes',
 'advertisement_total_bytes',
 'system_total_bytes',
 'voip_total_bytes',
 'speedtest_total_bytes',
 'email_total_bytes',
 'weather_total_bytes',
 'media_total_bytes',
 'mms_total_bytes',
 'others_total_bytes']
test['total_bytes_mod'] = 0
for i in byte_cols:
    test['total_bytes'] += test[i]
test.total_bytes_mod = test.total_bytes/1024 #for GB/s
test.total_bytes_mod = (test.total_bytes_mod/(test.par_min*60))


# In[240]:


test['cell_name'][1:10]


# In[241]:





# In[232]:


import datetime
temp =[]
for i in range(len(test)):
   temp.append(datetime.datetime(test['par_year'][i],test['par_month'][i],test['par_day'][i]).weekday())
test["week_day"] = temp


# In[233]:


test_x=test.drop(['cell_name','par_year','par_month','par_day','Congestion_Type'],axis=1)


# In[328]:
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(test_x['Congestion_Type'].astype(str))
test_x['Congestion_Type'] = mvar47
pred1_test = regressor.predict(test_x)
#pred1_test = xgb1.predict(test_x)


# In[329]:


len(pred1_test[pred1_test == 1])


# In[330]:


l = []
for i in range(len(test_x)):
    if pred1_test[i] == 1:
        l.append(test['cell_name'][i])


# In[331]:


df = pd.DataFrame(l)


# In[332]:


df = df.rename({0 : 'cell_name'},axis = 1)


# In[333]:


df['Predictions'] = 1


# In[334]:


df.head()


# In[335]:


dfnew = pd.merge(test,df)


# In[336]:


dfnew.shape


# In[337]:




# In[338]:


test.head()


# In[339]:


l2 = []
for i in range(len(test_x)):
    if pred1_test[i] == 0:
        l2.append(test['cell_name'][i])


# In[ ]:





# In[268]:


len(l2)


# In[340]:


new_test = test[pred1_test == 0]


# In[341]:


new_test.head()


# In[342]:


newnew_test = new_test.drop(['cell_name','par_year','par_month','par_day'],axis=1)


# In[343]:

pred2_test = regressor2.predict(newnew_test)
#pred2_test = xgb2.predict(newnew_test)


# In[344]:


pred2_test.shape


# In[345]:

new_test['Predictions'] = 0
new_test['Predictions'][pred2_test == 0] = 0
new_test['Predictions'][pred2_test == 1] = 2
new_test['Predictions'][pred2_test == 2] = 3


# In[346]:


new_test.head()


# In[347]:



dfnew['Predictions'] = 3


# In[348]:


dfnew.head()


# In[349]:


new_test['Predictions'][new_test['Predictions'] == 2] = 1
new_test['Predictions'][new_test['Predictions'] == 3] = 2


# In[350]:


new_test.head()


# In[351]:


iopp = pd.concat([dfnew,new_test])


# In[352]:


iopp.shape


# In[353]:


y_test = pd.read_csv('/home/kartik/Hall_data_2019/y_test.csv')


# In[354]:


iopp = iopp.sort_values(by = 'cell_name')


# In[355]:


y_test =  y_test.sort_values(by = 'cell_name')


# In[358]:


iopp.head()


# In[360]:


y_test.head()


# In[361]:


zz = y_test['Congestion_Type']


# In[ ]:





# In[302]:


zz


# In[362]:


le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(zz.astype(str))
zz = mvar47


# In[363]:


zz


# In[364]:


zz[0:20]


# In[365]:


matthews_corrcoef(zz,iopp['Predictions'])


# In[ ]:
