import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import numpy as np
from luminol.anomaly_detector import AnomalyDetector
from feature_selector import FeatureSelector
from sklearn.ensemble import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def clean_anomaly(df, index_name="15分钟段", var_name="实际功率", limit=0.5):
    df_clean = []
    for g_name, g in df.groupby(index_name):
        temp = deepcopy(g).reset_index(drop=True)
        limit_low, limit_up = np.percentile(temp[var_name], [5, 95])
        temp = temp[(temp[var_name] < limit_up) & (temp[var_name] > limit_low)].reset_index(drop=True)
        ts = temp[var_name]
        ts_mean = np.mean(ts)
        ts_std = np.std(ts)
        ts = (ts - ts_mean) / ts_std
        if ts_std > 0:
            my_detector = AnomalyDetector(ts.to_dict(), algorithm_name='exp_avg_detector')
            score = my_detector.get_all_scores()
            df_clean.append(temp[np.array(score.values) < limit])
        else:
            df_clean.append(temp)
    df_clean = pd.concat(df_clean, ignore_index=True)
    return df_clean


result = []
p_all = [20, 30, 10, 20, 21, 10, 40, 30, 50, 20]
for num, p in zip(range(1, 11), p_all):
    print(num)
    # 训练集数据整理
    train = pd.read_csv("./data/train/train_%s.csv" % num)
    # 官方提供清洗要求
    train = train[train["实际辐照度"] >= 0].drop("实际辐照度", axis=1)
    train["时间"] = pd.to_datetime(train["时间"], format='%Y/%m/%d %H:%M')
    if num == 7:
        train = train[(train["时间"] < "2018/03/01 00:00") | (train["时间"] > "2018/04/04 23:45")]
    if num == 9:
        train = train[(train["时间"] < "2016/01/01 9:00") | (train["时间"] > "2017/03/21 23:45")]
    train["15分钟段"] = train["时间"].dt.time
    plt.figure(figsize=[72, 8])
    sns.boxplot(x="15分钟段", y="实际功率", data=train)
    plt.savefig("./plot/boxplot_%02d.png" % num)
    train = clean_anomaly(train, index_name="15分钟段", var_name="实际功率", limit=1.5)
    plt.figure(figsize=[72, 8])
    sns.boxplot(x="15分钟段", y="实际功率", data=train)
    plt.savefig("./plot/boxplot_clear_%02d.png" % num)
    plt.close('all')
    # 处理测试集数据
    test = pd.read_csv("./data//test/test_%s.csv" % num)
    test["时间"] = pd.to_datetime(test["时间"], format='%Y/%m/%d %H:%M')
    test["15分钟段"] = test["时间"].dt.time
    # 处理天气数据
    weather = pd.read_csv("./data/气象数据/电站%s_气象.csv" % num, encoding="gbk")
    weather["时间"] = pd.to_datetime(weather["时间"], format='%Y/%m/%d %H:%M')
    # 合并所有数据
    data = pd.concat([train, test], ignore_index=True, sort=False)
    # 转换15分钟段数据
    data = data.merge(weather[["时间", "直辐射"]], left_on=["时间"], right_on=["时间"], how="left")
    period = data.groupby("15分钟段", as_index=False)["实际功率"].agg({"base_mean": "median"}).reset_index()
    period["15分钟段_num"] = np.abs(period["index"] - period["base_mean"].values.argmax())
    period = period.fillna(0)
    data = data.merge(period[["15分钟段", "15分钟段_num", "base_mean"]], left_on="15分钟段", right_on="15分钟段")
    # 构造新变量
    var_all = ['辐照度', '风速', '风向', '温度', '湿度', '压强', '直辐射', '15分钟段_num']
    for var1 in ['辐照度', '风速', '风向', '温度', '湿度', '压强', 'base_mean', '15分钟段_num']:
        for var2 in ['辐照度', 'base_mean', '15分钟段_num', '直辐射']:
            data["%s_%s" % (var1, var2)] = (data[var1] * data[var2])
    not_x = ['15分钟段', 'id', '时间', '实际功率']
    x_train = data[np.isnan(data.id)].drop(not_x, axis=1)
    y_train = data[np.isnan(data.id)].实际功率
    x_test = data[data.id > 0].drop(['15分钟段', 'id', '时间', '实际功率'], axis=1)
    fs = FeatureSelector(data=x_train, labels=y_train)
    fs.identify_collinear(correlation_threshold=0.99)
    choose = fs.ops['collinear']
    x_train_select = x_train.drop(choose, axis=1)
    x_test_select = x_test.drop(choose, axis=1)
    y_predict = []
    for var1 in ['辐照度', 'base_mean', '15分钟段_num', '直辐射', '']:
        print(var1)
        for var2 in x_test_select.columns:
            if var1 not in var2:
                x_train = x_train_select.drop(var2, axis=1)
                x_test = x_test_select.drop(var2, axis=1)
        x_all = pd.concat([x_train, x_test], ignore_index=True)
        x_std_tools = MinMaxScaler(feature_range=(-1, 1))
        x_std_tools.fit(x_all)
        x_all = x_std_tools.transform(x_all)
        x_train_std = x_std_tools.transform(x_train)
        x_test_std = x_std_tools.transform(x_test)
        pca = PCA(n_components=8)
        pca.fit(x_all)
        x_train_pca = pca.transform(x_train_std)
        x_test_pca = pca.transform(x_test_std)
        m1 = BaggingRegressor(LinearRegression(), n_estimators=100, n_jobs=5)
        m2 = AdaBoostRegressor(LinearRegression(), n_estimators=100)
        m3 = BaggingRegressor(DecisionTreeRegressor(min_samples_split=500),
                              n_estimators=100, n_jobs=5)
        m4 = AdaBoostRegressor(DecisionTreeRegressor(min_samples_split=500),
                              n_estimators=100)
        m5 = BaggingRegressor(DecisionTreeRegressor(min_samples_split=500), n_estimators=100, n_jobs=5)
        m6 = AdaBoostRegressor(DecisionTreeRegressor(min_samples_split=500), n_estimators=100)
        m7 = BaggingRegressor(ExtraTreeRegressor(min_samples_split=500), n_estimators=100, n_jobs=5)
        m8 = AdaBoostRegressor(ExtraTreeRegressor(min_samples_split=500), n_estimators=10)
        m9 = RandomForestRegressor(n_estimators=100, n_jobs=5,  min_samples_split=500)
        m10 = GradientBoostingRegressor(n_estimators=100,  min_samples_split=500)
        m = VotingRegressor([('m1', m1), ('m2', m2), ('m3', m3), ('m4', m4), ('m5', m5),
                             ('m6', m6), ('m7', m7), ('m8', m8), ('m9', m9), ('m10', m10)], n_jobs=5)
        m.fit(x_train_pca, y_train)
        test_predict = m.predict(x_test_pca)
        y_predict.append(test_predict)
    y_predict = np.vstack(y_predict)
    y_predict = np.median(y_predict, axis=0)
    plt.close('all')
    plt.figure()
    pd.Series(y_predict).plot()
    plt.savefig("./plot/predict_%02d.png" % num)
    temp_predict = pd.DataFrame({"id": data[data.id > 0].id.values,
                                 "prediction": y_predict})
    print(temp_predict.describe())
    result.append(temp_predict)
    temp_predict.to_csv("./predict/base_%s.csv" % num, index=False)
result = pd.concat(result, ignore_index=True)
result.id = result.id.astype(np.int)
result = result.sort_values(by="id").drop_duplicates()
result.to_csv("base.csv", index=False)
