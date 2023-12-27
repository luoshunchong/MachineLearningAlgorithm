import numpy as np
import random
import matplotlib.pyplot as plt


# 质心集：数组；
# 数据集：列表；
# 簇：字典

def loadDataSet(filename):
    '''

    input:
    return:
    '''
    data_get = []
    fp = open(filename, 'r')
    for line in fp:
        curline = line.split(' ')
        floatline = list(map(float, curline))
        data_get.append(floatline)
    return data_get


def findCentroids(data_get, k):
    '''
    获取k个质心
    input:
    return:
    '''
    m = random.sample(data_get, k)
    return np.array(m)


def distEclud(vecA, vecB):  # 计算距离--欧式距离
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def Kmeans(data_get, centroidsList):  # 划分簇
    #    np.array(data_get)     #将数据集转为数组
    distculde = {}  # 建立一个字典
    flag = 0  # 元素分类标记，记录与相应聚类距离最近的那个类
    for data in data_get:
        vecA = np.array(data)
        minDis = float('inf')  # 始化为最大值
        for i in range(len(centroidsList)):
            vecB = centroidsList[i]
            distance = distEclud(vecA, vecB)  # 计算距离
            if distance < minDis:  # 直至找出距离最小的质点
                minDis = distance
                flag = i
        if flag not in distculde.keys():
            distculde[flag] = list()
        distculde[flag].append(data)
    return distculde


def getCentroids(distculde):  # 得到新的质心
    newcentroidsList = []  # 建立新质点集
    for key in distculde:
        cent = np.array(distculde[key])  # 将列表转为数组，便于计算
        newcentroid = np.mean(cent, axis=0)  # 计算新质点，对x和y分别求和，再平均
        newcentroidsList.append(newcentroid.tolist())  # 添加每个质心，得从数组转化为列表添加
    return np.array(newcentroidsList)  # 返回新质点数组


def calculate_Var(distculde, centroidsList):
    # 计算聚类间的均方误差
    item_sum = 0.0
    for key in distculde:
        vecA = centroidsList[key]
        dist = 0.0
        for item in distculde[key]:
            vecB = np.array(item)
            dist += distEclud(vecA, vecB)
        item_sum += dist
    return item_sum


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


def main():
    k = 3
    datMat = list(np.load('data.npy'))
    centroidsList = findCentroids(datMat, k)  # 随机获得k个聚类中心
    distculde = Kmeans(datMat, centroidsList)  # 第一次聚类迭代
    newVar = calculate_Var(distculde, centroidsList)
    oldVar = -0.0001  # 初始化均方误差
    while abs(newVar - oldVar) >= 0.0001:
        centroidsList = getCentroids(distculde)
        distculde = Kmeans(datMat, centroidsList)
        oldVar = newVar
        newVar = calculate_Var(distculde, centroidsList)
    showCluster(distculde, centroidsList)


if __name__ == "__main__":
    main()
