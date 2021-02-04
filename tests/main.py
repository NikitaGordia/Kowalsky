from kowalsky.optuna import optimize
from kowalsky.optuna import models
from kowalsky.optuna import optimize_super_learner
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

# for model in models:
#     if model[-1] == 'C': continue
#     print(model)
#     optimize(model,
#              path='./feed_baseline.csv',
#              y_label='count',
#              direction='minimize',
#              scorer='rmsle',
#              trials=4)

# optimize_super_learner(models={
#     'LGBR': LGBMRegressor(
#         **{'learning_rate': 0.013264625110413585, 'n_estimators': 406, 'max_depth': 25, 'num_leaves': 1253,
#            'min_child_samples': 10}),
#     'XGBR': XGBRegressor(
#         **{'learning_rate': 0.006438488538697672, 'n_estimators': 579, 'max_depth': 12, 'gamma': 0.9267817436177146})
# }, head_models={
#     'LogReg': LogisticRegression()
# }, path='./feed_baseline.csv', y_label='count', direction='minimize', scorer='rmsle')

# model = LGBMRegressor(
#         **{'learning_rate': 0.013264625110413585, 'n_estimators': 406, 'max_depth': 25, 'num_leaves': 1253,
#            'min_child_samples': 10})
# import pandas as pd
# from sklearn.model_selection import train_test_split
# ds = pd.read_csv('./feed_baseline.csv')
# X_ds, y_ds = ds.drop('count', axis=1), ds['count']
# X_train, X_val, y_train, y_val = train_test_split(X_ds, y_ds)
# # model.fit(X_train, y_train)
# print("OK")
#
# from mlens.ensemble import SuperLearner
# super_model = SuperLearner()
# super_model.add(model)
# super_model.fit(X_train, y_train)
#
# from kowalsky.colab import csv
#
# print(csv("1DRt7_skit13bYJV1jS9oN7Ag8v6M-l0Y"))


