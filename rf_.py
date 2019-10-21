import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv('train_upd.csv')
df=df.sample(frac=1).reset_index(drop=True)
df=df.drop(['cell_name','par_year','par_month'],axis=1)
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(df['ran_vendor'].astype(str))
df["ran_vendor"] = mvar47

le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(df['Congestion_Type'].astype(str))
df["Congestion_Type"] = mvar47
#
df['div']=0
i=0
for row in df['par_hour']:

    if(int(row)<=5):
        df.loc[i,'div']=1
    elif(int(row)<=10):
        df.loc[i,'div']=2
    elif(int(row)<=15):
        df.loc[i,'div']=3
    elif(int(row)<=20):
        df.loc[i,'div']=4
    else:
        df.loc[i,'div']=5
    i+=1




df=df.drop(['par_hour'],axis=1)
# df['sum']=0
# for i in range (8,34):
#     df['sum']+=df[df.columns[i]]
#
# df['sum']/=26
#
# li=[]
# for i in range (0,38):
#     if (i<8 or i>33):
#         li.append(i)
# print(li)
# dfn=df.drop(df.columns[li],axis=1)
# dfn['std']=0
# dfn['std']=df.std(axis=1)
# df['std']=0
# df['std']=dfn['std']
# print(df.head(3))
#
# c=Counter(df['Congestion_Type'])
# print(c)
# print(df.head()
x=df.drop(['Congestion_Type'],axis=1)
x = x.apply(pd.to_numeric)
scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
y=df['Congestion_Type']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.15,random_state=1)

# print(x.isnull().sum().sum())
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(train_x,train_y)
my_scorer = make_scorer(matthews_corrcoef)
scores = cross_val_score(clf, train_x, train_y, cv=5, verbose=1, scoring=my_scorer)
print(scores.mean())
