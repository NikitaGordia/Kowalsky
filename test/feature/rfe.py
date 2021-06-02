from kowalsky.feature import XRFE
from kowalsky.feature import XRFECV
from sklearn.feature_selection import RFE
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor


def test_xrfe():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    model = DecisionTreeRegressor(max_depth=4, random_state=1)
    rfe = XRFE(model, n_features_to_select=1)
    rfe.fit(X, y)
    print(rfe.ranking_)


def test_xrfecv():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    model = DecisionTreeRegressor(max_depth=4, random_state=1)
    rfe = XRFECV(model, n_features_to_select=1, scorer='neg_mean_squared_error')
    rfe.fit(X, y)
    print(rfe.ranking_)


def test_rfe():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    model = DecisionTreeRegressor(max_depth=4, random_state=1)
    rfe = RFE(model, n_features_to_select=1)
    rfe.fit(X, y)
    print(rfe.ranking_)


if __name__ == '__main__':
    test_xrfecv()
