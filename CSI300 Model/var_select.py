import gplearn.fitness
import gplearn.functions
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import split
import gplearn 
from gplearn import genetic as gen
from gplearn.fitness import make_fitness
import tensorflow as tf
# helper functions for selecting and generating indicators
e= 1*10**-40

def gp_model(trainX, trainY, add_feat, parsimony, classification=False):
    '''Uses genetic programming to generate nonlinear combinations of features. 
    add_feat: number of features we want to generate
    parsimony: parsimony coefficient of gplearn model
    classification: Classification or regression model. True if we want classification
    returns: gp model'''

    dleaky_relu =gplearn.functions.make_function(function=leaky_relu,name='leaky_relu',arity=1)
    dtanh = gplearn.functions.make_function(function=tanh, name='tanh',arity=1)
    drelu =gplearn.functions.make_function(function=relu,name='relu',arity=1)
    # if classification:
    #     loss =make_fitness(function=binary_loss, greater_is_better=False)
    # else: 
    loss = 'spearman'
    function_set = ['add', 'sub', 'mul', 'div','log', 'neg','max',
                    'min','inv','tan',dleaky_relu, dtanh,drelu]
    gp = gen.SymbolicTransformer(generations=30, population_size=8000,
                         hall_of_fame=70, n_components=add_feat,
                         metric=loss,parsimony_coefficient= parsimony,
                         function_set=function_set, tournament_size=50,
                         max_samples=0.9, verbose=1,
                         random_state=0)
    gp.fit(trainX, trainY)
    return gp

def binary_loss(y, y_pred, w):
    '''Uses sigmoid activation and to create predictions
    y: actual value
    y_pred: predicted values
    w: weights'''
    y_pred = [tf.nn.sigmoid(float(x)) for x in y_pred]
    y_pred = np.asarray(y_pred)
    loss = -1*(y*np.log(y_pred+e)+(np.ones_like(y)-y)*np.log(np.ones_like(y_pred)-y_pred+e))
    return np.average(loss, weights=w)

def leaky_relu(array):
    return np.asarray([0.2*data if data<0 else data for data in array])

def relu(x):
    return np.asarray([0 if data<0 else data for data in x])

def tanh(x):
    return np.tanh(x)

def filter_params(input, labels, epochs=3, drop_rate=0.1, min_importance=0, is_split= False, obj='regression'):
    '''takes in a pd dataframe as input and array as label. Drops the features of low importance.
    drop_rate: percentage of the labels dropped in each epoch. 
    min_importance: the minimum importance a feature has to have
    is_split: uses importance by number of splits if True. Uses importance by gain if False.
    obj: objective of lgb model
    Returns: a list of names of column names'''
    params = {
        "boosting_type": "gbdt",
        "objective": obj,
        "n_estimators": 45,
        "learning_rate": 0.025,
        "bagging_fraction":0.9,
        "feature_fraction":0.7,
        "bagging_freq":8,
        "verbose":-1
    }
    trainX, testX, trainY, testY=split(input, labels, 0.2)
    prev_mse = 10**50

    for i in range(epochs):
        
        # train model and get the importance
        model = lgb.train(params, lgb.Dataset(trainX, trainY))
        pred = model.predict(testX)
        if is_split:
            importance = model.feature_importance()
        else:
            importance = model.feature_importance(importance_type="gain")

        # deletes the features that are not important enough. If the deletion causes 
        # a large decrease in accuracy then loop stops and previous iteration's features are returned
        # If not, it continues
        sorted = np.sort(importance)
        threshold = sorted[int(drop_rate*len(sorted))]
        cur_mse = mean_squared_error(testY, pred)
        delete = []

        if cur_mse<prev_mse:
            prev_mse = mean_squared_error(testY, pred)
            for j in range(len(importance)):
                if (importance[j] <=np.max((min_importance,threshold))):
                    delete.append(trainX.columns.values[j])
            if trainX.drop(delete,axis=1).empty:
                return trainX.columns.values.tolist()
            
            trainX = trainX.drop(delete,axis=1)
            testX = testX.drop(delete,axis=1)

        else:
            return trainX.columns.values.tolist()
        # delete all the features that are not important
    return trainX.columns.values.tolist()
