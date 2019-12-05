from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.linear_model import *
from xgboost import *
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor


def stack_base(x_train, y_train, x_test, pca_choose):
    if pca_choose:
        pca = TruncatedSVD(n_components=3, n_iter=100)
        pca.fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)
    else:
        x_train_pca = np.array(x_train)
        x_test_pca = np.array(x_test)

    dt = DecisionTreeRegressor(max_depth=6, min_samples_split=0.005, min_samples_leaf=0.005)
    ex_tree = ExtraTreesRegressor(max_depth=6, min_samples_split=0.005, min_samples_leaf=0.005,
                                  n_estimators=120, n_jobs=-1)
    rf = RandomForestRegressor(max_depth=6, min_samples_split=0.005, min_samples_leaf=0.005,
                               n_estimators=120, n_jobs=-1)
    xg = XGBRegressor(objective='reg:squarederror', max_depth=6, min_samples_split=0.005,
                      min_samples_leaf=0.005, n_estimators=30, n_jobs=-1)
    gbdt = GradientBoostingRegressor(max_depth=6, min_samples_split=0.005,
                                     min_samples_leaf=0.005, n_estimators=30)
    m_all = [

        ("LinearSVR", LinearSVR(max_iter=100)),

        ("SVR_linear", SVR(kernel="linear", max_iter=100)),

        ("SVR_rbf", SVR(kernel="rbf", max_iter=100)),

        ("Bayes", BayesianRidge(n_iter=100)),
        ('lr', LinearRegression()),

        ("Lasso", Lasso(max_iter=100)),

        ('mlp1', MLPRegressor(hidden_layer_sizes=10)),
        ('mlp2', MLPRegressor(hidden_layer_sizes=[3, 3])),
    ]
    m = VotingRegressor(m_all, n_jobs=4)
    m.fit(x_train_pca, y_train)
    train_predict = m.transform(x_train_pca)
    test_predict = m.transform(x_test_pca)
    return train_predict, test_predict
