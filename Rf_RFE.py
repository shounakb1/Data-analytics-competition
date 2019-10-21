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
from sklearn.feature_selection import RFE

df=pd.read_csv('train_upd.csv')
df=df.sample(frac=1).reset_index(drop=True)
df=df.drop(['cell_name','par_year','par_month'],axis=1)
le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(df['ran_vendor'].astype(str))
df["ran_vendor"] = mvar47

le = preprocessing.LabelEncoder()
mvar47 = le.fit_transform(df['Congestion_Type'].astype(str))
df["Congestion_Type"] = mvar47

x=df.drop(['Congestion_Type'],axis=1)

x = x.apply(pd.to_numeric)

x=(x-x.mean())/x.std()
y=df['Congestion_Type']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.15,random_state=1)

# rfe = RFE(clf, 1)
# rfe = rfe.fit(train_x, train_y)
# print(rfe.support_)
# print(rfe.ranking_)
rankings=[35, 29, 30, 31, 21, 27, 23, 26, 22, 8,  2,  1, 24,  6,  7,  3,  9 , 4 ,18,  5, 14, 25, 19, 12,
 13, 16, 10, 11, 20, 15, 17, 28, 32, 33, 34]
ranking_columns = []
for i in range(1,36):
    ranking_columns.append(x.columns[rankings.index(i)])

for i in range(10,36):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(train_x[ranking_columns[:i]],train_y)
    my_scorer = make_scorer(matthews_corrcoef)
    scores = cross_val_score(clf, train_x, train_y, cv=5, verbose=1, scoring=my_scorer)
    print(scores.mean())
    # scores = cross_val_score(rfe, train_x, train_y, cv=5, verbose=1, scoring=my_scorer)

    # xgb1.fit(X_train[ranking_columns[:i]], X_train["Congestion_Type"])
    # print(xgb1.score(X_test[ranking_columns[:i]],X_test["Congestion_Type"]))
