import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd

# Utilities

def split(inputs, labels, split):
    '''function that takes in inputs and labels. 
    returns: in order of lower split of input,
    higher split of input, lower split of label, upper split of label. The split is the ratio upper/lower '''
    lower_splitX = inputs[:int(len(inputs)*(1-split))+1]
    lower_splitY = labels[:int(len(inputs)*(1-split))+1]
    upper_splitX = inputs[int(len(inputs)*(1-split))+1:]
    upper_splitY = labels[int(len(inputs)*(1-split))+1:]
    return lower_splitX, upper_splitX, lower_splitY, upper_splitY

def obj( trainY, valY, trainX,valX,objective, args):
    '''Objective function used for hyperparameter tuning. 
    args: a set of the hyperparameters {n_estimators, learning_rate}. 
    returns: MSE of the model trained on the training set, and tested on the validation set'''
    params =  {
    "boosting_type": 'gbdt',
    "objective": objective,
    "learning_rate": args['learning_rate'],
    "feature_fraction": args['feature_fraction'],
    # "bagging_fraction":0.9,
    # "bagging_freq":8,
    # "max_depth":args['max_depth'],
    "verbose":-1}

    model = lgb.train(params, lgb.Dataset(pd.DataFrame(trainX),pd.DataFrame(trainY)),
                      num_boost_round=args['n_estimators'])
    # print('validation loss', mean_squared_error(valY,model.predict(valX)))
    # print('learning',args['learning_rate'])
    return mean_squared_error(valY,model.predict(valX))


