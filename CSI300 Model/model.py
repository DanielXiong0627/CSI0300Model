import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from var_select import gp_model
import warnings
from utils import split
from var_select import filter_params
warnings.filterwarnings(action='ignore', category=UserWarning)
import copy
from hypparam import hypmodel
# Model file; makes predictions on the percentage change from tomorrow to two days later based on
# data from today and before. Will create signal to be used in strategy testing. Make sure
# to put dates.csv, labels.csv, signal.csv into strategy_testing folder to test 

FILE_PATH = 'processedHS300.csv'
START_DATE = 100
END_DATE = 20
TIME_DIFF=0
data = pd.read_csv(FILE_PATH)

indicators = data[['aroon','dx', 'ema','CLOSE','HIGH', 'OPEN', 'LOW', 'atr','obv','kama','rsi',
                   'cci','ht_period', 'cmo','ht_trend','bop','macd', 'ht']][START_DATE:len(data)-END_DATE]
# target = ta.EMA(data['label'], TIME_DIFF)
target = data['label'][START_DATE:len(data)-END_DATE]
dates = data['date'][START_DATE:len(data)-END_DATE]
price = data['price'][START_DATE:len(data)-END_DATE]

default_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "n_estimators": 45,
    "learning_rate": 0.025,
    "bagging_fraction":0.8,
    "feature_fraction":0.7,
    "bagging_freq":8,
    "verbose":-1
}

test_no = 1 # number of days that the trained model is tested on
train_no = 1300
total_mse = 0
total_l1 = 0
val_split = 0.15
class_list = []
actual_list= []
price = price[train_no+1:]
dates = dates[train_no+1:]

# gp = gp_model(indicators[:train_no+1],target[:train_no+1], 20)
# sliding window that predicts the percentage change in price in the next day
# Classification
class_target = pd.DataFrame([1 if target.iloc[i] >=0 else 0 for i in np.arange(len(target))])
for i in range(len(target)-(test_no+train_no)):

    # split into train test validation
    trainX = indicators.copy()[i:i+train_no+1-2]
    trainY = class_target.copy()[i:i+train_no+1-2]
    trainX, valX, trainY, valY = split(trainX,trainY, val_split)
    testX = indicators.copy()[i+train_no+1:i+train_no+test_no+1]
    testY = class_target.copy()[i+train_no+1:i+train_no+test_no+1]

    # Use gp to create indicators at beginning 
    if i == 0:
        gp = gp_model(trainX,trainY, 10, 0.005, True)
    
    trainX.reset_index(drop=True, inplace=True)
    trainX = trainX.join(pd.DataFrame(gp.transform(trainX)))
    valX.reset_index(drop=True, inplace=True)
    valX = valX.join(pd.DataFrame(gp.transform(valX)))
    testX.reset_index(drop=True, inplace=True)
    testX = testX.join(pd.DataFrame(gp.transform(testX)))
    opt_features = filter_params(trainX,trainY,5,0.2,0,False,'binary') # function that takes the features with most predictive power
    print(opt_features)
    trainX = trainX[opt_features]
    testX = testX[opt_features]
    valX = valX[opt_features]
    
    # optimize hyperparameters
    if i%500==0:
        class_params = copy.deepcopy(default_params)
        class_params['objective'] = 'binary'
        class_params['boosting_type'] = 'rf'
        hmodel = hypmodel(trainY,valY,trainX,valX,class_params['boosting_type'],class_params['objective'])
        class_params['n_estimators'], class_params['learning_rate'], class_params['feature_fraction'] = hmodel.opt()
        
    # extract hyperparameters and train on best hyperparameters. Use it on test set.
    class_gbm = lgb.train(class_params, lgb.Dataset(trainX,trainY))
    pred = class_gbm.predict(testX)
    class_list.append(pred[0])
    actual_list.append(np.asarray(testY[testY.columns[0]])[0])

    # print statements and error calculations
    if i>0:
        print('loss',  log_loss(actual_list, class_list))
    print('trees',class_params['n_estimators'])
    print('bagging', class_params['bagging_fraction'])
    print('learning_rate',class_params['learning_rate'])
    print('progress', i+1,'/',len(target)-(test_no+train_no))

time = np.arange(len(target)-(test_no+train_no))

# Regression
reg_list = []
actual_list= []
for i in range(len(target)-(test_no+train_no)):

    # split into train test validation
    trainX = indicators.copy()[i:i+train_no+1-2]
    trainY = target.copy()[i:i+train_no+1-2]
    trainX, valX, trainY, valY = split(trainX,trainY, val_split)
    testX = indicators.copy()[i+train_no+1:i+train_no+test_no+1]
    testY = target.copy()[i+train_no+1:i+train_no+test_no+1]

    # Use gp to create indicators at beginning (or periodically for the line below)
    if i == 0:
        gp = gp_model(trainX,trainY, 10, 0.01)
    
    trainX.reset_index(drop=True, inplace=True)
    trainX = trainX.join(pd.DataFrame(gp.transform(trainX)))
    valX.reset_index(drop=True, inplace=True)
    valX = valX.join(pd.DataFrame(gp.transform(valX)))
    testX.reset_index(drop=True, inplace=True)
    testX = testX.join(pd.DataFrame(gp.transform(testX)))
    opt_features = filter_params(trainX, trainY,5,0.1,0, False) # function that takes the features with most predictive power
    trainX = trainX[opt_features]
    testX = testX[opt_features]
    valX = valX[opt_features]
    print(trainX.columns.values)

    # optimize hyperparameters
    if i%500==0:
        hmodel = hypmodel(trainY,valY,trainX,valX, 'gbdt','regression')
        reg_params = copy.deepcopy(default_params)
        reg_params['n_estimators'],reg_params['learning_rate'], reg_params['feature_fraction'] = hmodel.opt()
        
    # extract hyperparameters and train on best hyperparameters. Use it on test set.
    reg_gbm = lgb.train(reg_params,lgb.Dataset(trainX,trainY))
    pred = reg_gbm.predict(testX)
    reg_list.append(pred[0])
    actual_list.append(testY)

    # print statements and error calculations
    total_mse += mean_squared_error(testY, pred)
    total_l1 += sk.mean_absolute_error(testY, pred)
    # print('feature', reg_params['feature_fraction'])
    # print('trees',reg_params['n_estimators'])
    # print('learning_rate',reg_params['learning_rate'])
    print('r2', 1-(total_mse/(i+1))/mean_squared_error(actual_list,np.ones_like(actual_list)*np.mean(actual_list)))
    print('progress', i+1,'/',len(target)-(test_no+train_no))
print('averageMSE', total_mse/(len(target)-(test_no+train_no)))
print('l1loss', total_l1/(len(target)-(test_no+train_no)))
print('mseofmean', mean_squared_error(actual_list,np.ones_like(actual_list)*np.mean(actual_list)))

# visualization
time = np.arange(len(target)-(test_no+train_no))
plt.plot(time, actual_list, label='actual')
plt.plot(time, reg_list, label='predicted')
plt.legend()
plt.show()
# plt.plot(np.arange(len(target)), target)

# save the label and predicted. Drag all of these files into strategy testing folder if results are satisfactory
pd.DataFrame(price[1:]).to_csv('labels.csv')
signal = pd.DataFrame({'reg':reg_list[:-1], 'class':class_list[:-1]})
signal.to_csv('signal.csv')
pd.DataFrame(dates[1:]).to_csv('dates.csv')