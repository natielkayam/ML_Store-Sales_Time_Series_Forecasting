import pandas as pd
import numpy as np

#OIL
oil = pd.read_csv("Data/oil.csv", low_memory= False)
#TRANSACTIONS
transactions = pd.read_csv("Data/transactions.csv", low_memory= False)
#HOLIDAYS
holidays = pd.read_csv("Data/holidays_events.csv", low_memory= False)
#TRAIN
train = pd.read_csv("Data/train.csv" , low_memory= False)
#TEST
test = pd.read_csv("Data/test.csv" , low_memory= False)
#STORES
stores = pd.read_csv("Data/stores.csv" , low_memory= False)

#connect all
train['test'] = 0
test['test'] = 1

#add all the features also to the test so we wont have to handle it later
data = pd.concat([train, test], axis=0)
data = data.merge(holidays, on='date', how='left')
data= data.merge(stores, on='store_nbr', how='left')
data= data.merge(oil, on='date', how='left')
data= data.merge(transactions, on=['date', 'store_nbr'], how='left')
data = data.set_index(['store_nbr', 'date', 'family'])
data = data.drop(index='2013-01-01', level=1)

data_ = data.copy().reset_index()
data_.info()

#split to train / test
train = data_[data_['test'] == 0]
test = data_[data_['test'] == 1]

train['date'] = pd.to_datetime(train["date"])
train['day_of_week'] = train['date'].dt.day_of_week
train['day_of_year'] = train['date'].dt.dayofyear
train['day_of_month'] = train['date'].dt.day
train['month'] = train['date'].dt.month
train['quarter'] = train['date'].dt.quarter
train['year'] = train['date'].dt.year


#calculate sum sales for every day of the month
grouping_columns = ['year', 'quarter', 'month', 'day_of_week', 'day_of_year', 'day_of_month']
for ind, column in enumerate(grouping_columns):
    grouped_data = train.groupby(column)['sales'].sum()
    grouped_data = pd.DataFrame(grouped_data).reset_index()

#calculate lags for each store_nbr and family of products
data_ = data.copy().reset_index()
grouped_data = data_.groupby(['store_nbr', 'family'])
alphas = [0.95, 0.8, 0.65, 0.5]
lags =[1,7,30]
for a in alphas:
    for i in lags:
        data_[f'sales_lag_{i}_alpha_{a}'] = np.log1p(grouped_data['sales'].transform(lambda x: x.shift(i).ewm(alpha=a, min_periods=1).mean()))


data_['date'] = pd.to_datetime(data_["date"])
data_['day_of_week'] = data_['date'].dt.day_of_week
data_['day_of_year'] = data_['date'].dt.dayofyear
data_['day_of_month'] = data_['date'].dt.day
data_['month'] = data_['date'].dt.month
data_['quarter'] = data_['date'].dt.quarter
data_['year'] = data_['date'].dt.year
data_['family'] = data_['family'].astype('category').cat.codes
data_['city'] = data_['city'].astype('category').cat.codes
data_['state'] = data_['state'].astype('category').cat.codes

#convert promotions to binary
data_['onpromotion'] = data_['onpromotion'].apply(lambda x: x > 0)

#get lags columns
sales_lag_columns = list(data_.filter(like="lag").columns)

#convert to dummises
to_dummies = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'store_nbr', 'cluster', 'family', 'onpromotion',
       'locale', 'locale_name', 'city', 'state']

#create X to model
X = data_.loc[:, [ 'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'store_nbr', 'cluster', 'family', 'onpromotion',
       'locale', 'locale_name',  'city', 'state', 'test', 'sales', 'id']+ sales_lag_columns]

X = X.fillna(0)
X[to_dummies] = X[to_dummies].astype('category')
X[to_dummies] = X[to_dummies].apply(lambda x: x.cat.codes)

data_train = X[X['test'] == 0]

import DECISIONTREE
#dtmodel = dt(data_train)
# xgModel = xgboost(data_train)
import KNN
knnModel = activeKNN(data_train)
#--------------------------

data_test = X[X['test'] == 1]
data_test_id = data_test['id']
data_test = data_test.drop(columns = {'sales','test','id'})
#y_pred = dtmodel.predict(data_test)
#y_pred = xgModel.predict(data_test)
y_pred = knnModel.predict(data_test)

output = pd.DataFrame(index=data_test_id)
output['sales'] = y_pred
output['sales'] = output['sales'].clip(0)

output.to_csv('newsubmissionKNN.csv')

