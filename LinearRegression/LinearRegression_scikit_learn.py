import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""线性回归预测房子价格"""
# 获取数据
dataMat = []
labelMat = []
fr = open("ex1.txt")
for line in fr.readlines():
    curLine = line.strip().split('\t')
    dataMat.append(float(curLine[1]))
    labelMat.append(float(curLine[2]))

# 分割数据集 （训练集、测试集）
x_train, x_test, y_train, y_test = train_test_split(np.array(dataMat), np.array(labelMat), test_size=0.25)
# 进行标准化
# 特征值 目标值 都标准化
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train.reshape(-1, 1))
x_test = std_x.fit_transform(x_test.reshape(-1, 1))

std_y = StandardScaler()
y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 传入数据必须二维
y_test = std_y.fit_transform(y_test.reshape(-1, 1))

# estimator预测
# 正规方程求解方式
print("正规方程求解方式")
lr = LinearRegression()
lr.fit(x_train, y_train)

# print(lr.coef_)
# inverse_transform(X_scaled)是将标准化后的数据转换为原始数据。
y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
# print("预测：\n", y_lr_predict)
print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
print(f"R : {lr.score(x_test, y_test)}")

# 梯度下降方式
print("SGDRegressor度下降方式")
sgd = SGDRegressor()
sgd.fit(x_train, y_train.ravel())  # 为什么使用.ravel()：数据转换警告：当需要一维数组时，传递了列向量y。请将Y的形状更改为（n_samples，），例如使用ravel（）。

y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test).reshape(-1, 1))
# print(y_sgd_predict)
print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
print(f"R : {sgd.score(x_test, y_test)}")
