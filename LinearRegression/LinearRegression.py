# 导入用到的包
import numpy as np
import matplotlib.pyplot as plt


# 线性回归类
class LinerRegression():

    def __init__(self, learning_rate=0.01, max_iter=100, seed=None):
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.a = np.random.normal(1, 0.1)
        self.b = np.random.normal(1, 0.1)
        self.loss_arr = []

    def fit(self, x, y):
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._train_step()
            self.loss_arr.append(self.loss())

    def _f(self, x, a, b):
        """一元线性函数"""
        return x * a + b

    def predict(self, x=None):
        """预测"""
        if x is None:
            x = self.x
        y_pred = self._f(x, self.a, self.b)
        return y_pred

    def loss(self, y_true=None, y_pred=None):
        """损失"""
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict(self.x)
        return np.mean((y_true - y_pred) ** 2)

    def _calc_gradient(self):
        """梯度"""
        d_a = np.mean((self.x * self.a + self.b - self.y) * self.x)
        d_b = np.mean(self.x * self.a + self.b - self.y)
        print(d_a, d_b)
        return d_a, d_b

    def _train_step(self):
        """训练频度"""
        d_a, d_b = self._calc_gradient()
        self.a = self.a - self.lr * d_a
        self.b = self.b - self.lr * d_b
        return self.a, self.b


def sqrt_R(y_pred, y_test):
    y_c = y_pred - y_test
    rss = np.mean(y_c ** 2)
    y_t = y_test - np.mean(y_test)
    tss = np.mean(y_t ** 2)
    return (1 - rss / tss)


def makedata():
    # 生成数据
    # 设置随机生成算法的初始值
    np.random.seed(300)
    data_size = 100
    #  生出size个符合均分布的浮点数，取值范围为[low, high)，默认取值范围为[0, 1.0)
    x = np.random.uniform(low=1.0, high=10.0, size=data_size)
    y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)

    # 查看数据
    print(x)
    print(y.shape)
    plt.scatter(x, y)
    plt.show()

    # 区分训练集和测试集
    # 返回一个随机排列
    split_index = int(data_size * 0.75)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test


def build(x_train, y_train, x_test, y_test):
    # 训练数据
    regr = LinerRegression(learning_rate=0.01, max_iter=10, seed=314)
    regr.fit(x_train, y_train)

    # 可视化
    print(f'cost: \t{regr.loss():.3}')
    print(f"f={regr.a:.2} x + {regr.b:.2}")
    plt.scatter(np.arange(len(regr.loss_arr)), regr.loss_arr, marker='o', c='green')
    plt.show()

    # 评估
    y_pred = regr.predict(x_test)
    return y_pred


if __name__ == "__main__":
    '''
    采用梯度下降方法构建求解线性回归
    '''
    x_train, y_train, x_test, y_test = makedata()
    y_pred = build(x_train, y_train, x_test, y_test)
    r = sqrt_R(y_pred, y_test)
    print(f"R: {r}")
