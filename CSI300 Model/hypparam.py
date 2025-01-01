import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import pandas as pd
from functools import partial
from hyperopt import hp
from hyperopt import fmin, tpe


class hypmodel():
    '''model that optimizes hyperparameters given the training and validation set for lgb'''
    def __init__(self,trainY, valY, trainX,valX, type, objective) -> None:
        self.trainX= trainX
        self.trainY=trainY
        self.valY=valY
        self.valX=valX
        self.type = type
        self.objective = objective
        pass
    
    def obj(self, args):
        '''Objective function used for hyperparameter tuning. 
        args: a set of the hyperparameters {n_estimators, learning_rate, feature_fraction}. 
        returns: MSE of the model trained on the training set, and tested on the validation set'''

        params =  {
        "boosting_type": self.type,
        "objective": self.objective,
        "learning_rate": args['learning_rate'],
        "feature_fraction": args['feature_fraction'],
        "bagging_fraction":0.8,
        # "bagging_freq":8,
        # "max_depth":args['max_depth'],
        "verbose":-1}

        model = lgb.train(params, lgb.Dataset(pd.DataFrame(self.trainX),pd.DataFrame(self.trainY)),
                        num_boost_round=args['n_estimators'])
        # print('validation loss', mean_squared_error(valY,model.predict(valX)))
        # print('learning',args['learning_rate'])
        if self.objective == 'regression':
            return mean_squared_error(self.valY,model.predict(self.valX))
        if self.objective == 'binary':
            return log_loss(self.valY, model.predict(self.valX))
    
    def opt(self):
        '''finds optimal hyperparameters given the objective.
        Takes: objective (binary or regression)
        returns: int(best['n_estimators']), best['learning_rate'], best['feature_fraction'], best['bagging_fraction']
        '''
        fmin_func = partial(self.obj)
        space = {'n_estimators': hp.uniformint('n_estimators', 1, 200),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.9),
                'feature_fraction': hp.uniform('feature_fraction', 0.01,1),
                # 'bagging_fraction':hp.uniform('bagging_fraction', 0.1,0.9)
                }
        best = fmin(fmin_func,space,tpe.suggest, max_evals=50, verbose=-1)
        return int(best['n_estimators']), best['learning_rate'], best['feature_fraction']

