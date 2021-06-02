import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from kowalsky.tune import RandomSearchTuner
from kowalsky.tune import OptunaTuner
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from kowalsky.tune import optimize


def test_xrand_search():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    def objective(params):
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        return -mean_squared_error(y_test, model.predict(X_test))

    tuner = RandomSearchTuner(objective, {
        'max_depth': ('int', 2, 25),
        'min_samples_split': ('int', 2, 20),
        'min_weight_fraction_leaf': ('uniform', 0.0, 0.5),
        'min_samples_leaf': ('int', 1, 15)
    }, verbose=True)

    tuner.tune(100)
    print(tuner.best_score, tuner.best_params)

def test_optuna():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    def objective(params):
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        return -mean_squared_error(y_test, model.predict(X_test))

    tuner = OptunaTuner(objective, {
        'max_depth': ('int', 2, 25),
        'min_samples_split': ('int', 2, 20),
        'min_weight_fraction_leaf': ('uniform', 0.0, 0.5),
        'min_samples_leaf': ('int', 1, 15)
    })

    tuner.tune(100)
    print(tuner.best_score, tuner.best_params)

def test_optimize():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X['label'] = y
    optimize('dtR', 'neg_mean_squared_error', 'label', ds=X, tuner='optuna', show_live=False, trials=150, logging='local')

if __name__ == '__main__':
    test_optimize()
