import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def showCluster(distculde, centroidsList):
    # 画聚类图像
    colourList = ['bo', 'co', 'yo', 'go']
    for i in distculde:
        # 获取每簇
        centx = []
        centy = []
        for item in distculde[i]:
            centx.append(item[0])
            centy.append(item[1])
        plt.plot(centx, centy, colourList[i])  # 画簇
    # 画质心
    x = []
    y = []
    x.append(centroidsList[:, 0].tolist())
    y.append(centroidsList[:, 1].tolist())
    plt.plot(x, y, 'r^', markersize=10)  # 画质点，为红色三角号
    plt.show()


def kMenas():
    # 导入数据
    X = np.load("data.npy")
    # 使用sklearn构建模型
    model = KMeans(n_clusters=3, n_init='auto').fit(X)  # n_clusters指定3类，拟合数据
    centroids = model.cluster_centers_  # 聚类中
    label = model.labels_

    label_set = list(set(label))
    print(label)

    # 升维度
    labels = label[:, np.newaxis]
    # 标签和原始数据拼接
    X_lables = np.hstack((X, labels))
    # 构建字典
    distculde = {i: [] for i in label_set}
    for i in range(len(X_lables)):
        distculde[X_lables[i][2]].append([X_lables[i][0], X_lables[i][1]])
    # 画图
    showCluster(distculde, centroids)


if __name__ == "__main__":
    kMenas()
