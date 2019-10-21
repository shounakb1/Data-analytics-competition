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
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(train_data['ran_vendor'].astype(str))
train_data["ran_vendor"] = mvar47
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
x=train_data.drop(['Congestion_Type','cell_name','par_year','par_month','par_day','total_bytes'],axis=1)
y = train_data["Congestion_Type"]
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=20)
y1 = y_train["Congesti1on_Type"]
y1[y1 == "4G_BACKHAUL_CONGESTION"] = "C"
y1[y1 == "4G_RAN_CONGESTION"] = "C"
y1[y1== "3G_BACKHAUL_CONGESTION"] = "C"

le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(y1.astype(str))
y1 = mvar47

rom sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
regressor=RandomForestClassifier(n_estimators=700,max_features=3,max_depth=26,random_state=0,n_jobs=-1)
regressor.fit(x_train,y1)
new_x = x_train[y_train['Congestion_Type']!= "NC"]
new_y=y_train[y_train['Congestion_Type']!='NC']
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(new_y.astype(str))
new_y = mvar47
xgb2 = xgb.XGBClassifier(booster='gbtree', colsample_bylevel=1,
    colsample_bytree=1, gamma=0, learning_rate=0.15,
    max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
    n_estimators=700, n_jobs=3, nthread=1, objective='multi:softmax',
    random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
    seed=None, subsample=1)
xgb2.fit(new_x,new_y)

y1 = y_test["Congesti1on_Type"]
y1[y1 == "4G_BACKHAUL_CONGESTION"] = "C"
y1[y1 == "4G_RAN_CONGESTION"] = "C"
y1[y1== "3G_BACKHAUL_CONGESTION"] = "C"

le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(y1.astype(str))
y1 = mvar47
pred=regressor.predict(x_test)
new_x = x_test[y_test['Congestion_Type']!= "NC"]
new_y=y_test[y_test['Congestion_Type']!='NC']
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(new_y.astype(str))
new_y = mvar47

pred2=xgb2.predict(x_test)
