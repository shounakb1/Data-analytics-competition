from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

df=pd.read_csv('train_upd.csv')
df=df.sample(frac=1).reset_index(drop=True)
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(df['ran_vendor'].astype(str))
df["ran_vendor"] = mvar47


le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(df['Congestion_Type'].astype(str))
df["Congestion_Type"] = mvar47

import datetime
temp =[]
for i in range(len(df)):
   temp.append(datetime.datetime(df['par_year'][i],df['par_month'][i],df['par_day'][i]).weekday())
df["week_day"] = temp
df=df.drop(['cell_name','par_year','par_month','par_day'],axis=1)


x=df.drop(['Congestion_Type',],axis=1)

x = x.apply(pd.to_numeric)

x=(x-x.mean())/x.std()
from sklearn.decomposition import PCA

pca = PCA(n_components=15)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents)
y=df['Congestion_Type']
# train_x,test_x,train_y,test_y=train_test_split(principalDf,y,test_size=0.15,random_state=1)

clf=RandomForestClassifier(600,n_jobs=-1)
    # clf.fit(train_x[ranking_columns[:i]],train_y)
param_grid = {"max_depth": [10],
                  "max_features":[10]}
# ada = AdaBoostClassifier(n_estimators=10, base_estimator=clf,learning_rate=0.1)


# ada.fit(train_x,train_y)
my_scorer = make_scorer(matthews_corrcoef)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4,scoring=my_scorer,verbose=10,n_jobs = -1)
grid_search.fit(principalDf,y)
# scores = cross_val_score(clf, train_x, train_y, cv=4, verbose=1, scoring=my_scorer)
# print(scores)
# print(scores.mean())
