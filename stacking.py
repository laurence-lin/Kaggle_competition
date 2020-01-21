# Stacking ensemble implemented by: https://www.kaggle.com/mubashir44/simple-ensemble-model-stacking


class sklearnhelper(object):
  '''
  Helper that defined sklearn model in a general way, which we can stack them together for ensemble later
  '''
  def __init__(self, estimator, params = None, seed = 0):
    params['random_state'] = seed
    self.estimator = estimator(**params)

  def train(self, x_train, y_train):
    self.estimator.fit(x_train, y_train)

  def predict(self, x):
    return self.estimator.predict(x)

  def feature_importance(self, x, y):
    # print feature importance for model fit with x & y
    print(self.estimator.fit(x, y).feature_importances_)

n_folds = 5
kfold = sklearn.cross_validation.KFold(x_train, n_folds = 5, random_state = 9)

def get_oof(estimator, x_train, y_train, x_test):
  '''
  Train data on level 1 baseline model, and return hold-out set prediction for level 2 meta-model
  I don't know why predict x_test in this step
  x_train: train data
  x_test: test data
  '''
  oof_train = np.zeros((1, len(x_train))) # store predictions by base_model for train data
  oof_test = np.zeros((1, len(x_test)))  # overall test data prediction, made by the mean prediction of trained base model on each fold
  oof_test_skf = np.zeros((n_folds, len(x_test))) # test data prediction by trained base model, which is trained by each fold

  for i, (train_id, valid_id) in enumerate(kfold):
    x_tr = x_train[train_id]
    y_tr = y_train[train_id]
    x_oof = x_train[valid_id] # hold out data

    estimator.fit(x_tr, y_tr) # train on k-1 fold data
    oof_train[valid_id] = estimator.predict(x_oof) # combine each 1 fold hold-out data into full prediction of train data
    oof_test_skf[i, :] = estimator.predict(x_test)

  oof_test = oof_test_skf.mean(axis = 0) # mean prediction for each fold trained model

  return oof_train.reshape((-1, 1)), oof_test.reshape((-1, 1))


# Define base model: random forest, adaboost regressor, lightgbm regressor

rf_param = {
    'n_estimators':50,
    'max_depth':10,
    'verbose':0
}

seed = 9
rf = sklearnhelper(RandomForestRegressor(), rf_param, seed)

lgb_param = {
    'n_estimators':400,
    'num_leaves':40
}

lgbm = lgb.LGBMRegressor(lgb_param, random_state = seed)

# Stacking: level 1 base line model training
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
lgb_oof_train, lgb_oof_test = get_oof(lgbm, x_train, y_train, x_test)

# Stacking: level 2 meta model training
base_prediction = pd.DataFrame({'RandomForest':
                                rf_oof_train.ravel(),
                                })


# metao model input: concatenate the predictions from baseline model
x_meta = np.concatenate((rf_oof_train, lgb_oof_train), axis = 1) 
y_meta = y_train
meta_test = np.concatenate((rf_oof_test, lgb_oof_test), axis = 1)

meta_model = sklearn.linear_model.LinearRegression()
# Train meta model
meta_model.fit(x_meta, y_meta)



# Self wrapped up stacking model

# Cross validation stacking: 
class stacking(model):
  '''
  Train baseline model on K-1 Fold data, make predictions on out-of-fold data
  Assume model is sklearn type
  '''
  def __init__(self):
    self.model = model

  def train(self):
      prediction = []
      # CV training on level 1 
      for i, (train_id, valid_id) in enumerate(kfold):
          x_tr = x_train[train_id] # (k-1) fold training data
          y_tr = y_train[train_id]
          x_te = x_train[valid_id] # 1 hold-out-set data for prediction
          # Training on k-1 fold
          model.fit(x_tr, y_tr)
          # prediction on hold-out data
          pred_oof = model.predict(x_te)
          prediction.append(pred_oof)
     
      prediction = np.array(prediction)
      # Training for meta-model in level 2
      model.fit(prediction, y_train)

  def predict(self):
     
