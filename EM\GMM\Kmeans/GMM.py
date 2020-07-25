#! /usr/bin/env python
import numpy as np
import pandas as pd
from numpy import *
import pylab
import random, math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.K = n_clusters
        self.max_iter = max_iter

        self.posteriori = None      # 后验概率
        self.mu = None              # 均值
        self.cov = None             # 协方差矩阵
        self.priori = None          # 先验概率

    def init_kmeans(self, data):
        """
        采用kmeans算法对GMM参数进行初始化
        :param data: 输入数据点
        :return: 初始化后的参数
        """
        N, _ = data.shape
        k_means = KMeans(n_clusters=self.K)
        k_means.fit(data)
        category = k_means.labels_
        # 初始化posteriori, shape = (K, N)
        self.posteriori = np.zeros((self.K, N))
        # 初始化mu, shape = (3, 2)
        self.mu = k_means.cluster_centers_
        # 初始化cov, shape = (3, (2, 2))
        self.cov = np.asarray([np.cov(data[category == k], rowvar=False) for k in range(self.K)])
        # 初始化priori, shape = (3, 1)
        value_count = pd.Series(category).value_counts()
        self.priori = np.asarray([value_count[k] / N for k in range(self.K)]).reshape((self.K, 1))

    def ini_random(self, data):
        """
        随机对GMM参数进行初始化
        :param data: 输入数据点
        :return: 初始化后的参数
        """
        N, _ = data.shape
        self.posteriori = np.zeros((self.K, N))
        self.mu = data[np.random.choice(np.arange(N), self.K, replace=False)]
        self.cov = np.asarray([np.cov(data, rowvar=False)] * self.K)
        self.priori = np.ones((self.K, 1)) / self.K

    def get_expectation(self, data):
        """
        更新posteriori(后验概率）
        :param data: 输入的数据点
        :return: 更新后的后验概率
        """
        for k in range(self.K):
            # 计算高斯分布
            self.posteriori[k] = multivariate_normal.pdf(
                data,
                mean = self.mu[k],
                cov = self.cov[k]
            )
        # ravel(): 将多维数组转化为一维数组,  np.diag(): 将一维矩阵转化为对角矩阵
        self.posteriori = np.dot(np.diag(self.priori.ravel()), self.posteriori)
        self.posteriori /= np.sum(self.posteriori, axis=0)
    def get_mu(self):
        # 复制均值，中心点
        return np.copy(self.mu)
    def fit(self, data):
        N, _ = data.shape
        # 初始化GMM函数的参数
        self.init_kmeans(data)
        # 迭代
        for i in range(self.max_iter):
            # E-step: 跟新后验概率
            self.get_expectation(data)

            # get effective count 获得Nk的值
            effecitve_count = np.sum(self.posteriori, axis=1)

            # M-step: 更新GMM的参数
            self.mu = np.asarray([np.dot(self.posteriori[k], data) / effecitve_count[k] for k in range(self.K)])
            self.cov = np.asarray([np.dot((data - self.mu[k]).T, np.dot(np.diag(self.posteriori[k].ravel()), data - self.mu[k]))/ effecitve_count[k] for k in range(self.K)])
            self.priori = (effecitve_count / N).reshape((self.K, 1))
    def predict(self, data):
        self.get_expectation(data)
        result = np.argmax(self.posteriori, axis=0)
        return result
# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    K = 3
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=K)
    gmm.fit(X)
    category = gmm.predict(X)
    # print(cat)
    # 可视化
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]

    for k in range(K):
        # plt. scatter(X[:, 0], X[:, 1], c = category, label=labels[k])
        plt.scatter(X[category == k][:, 0], X[category == k][:, 1], c=color[k], label= labels[k])

    mu = gmm.get_mu()
    plt.scatter(mu[:, 0], mu[:, 1], s=200, c='grey', marker='P', label='Centroids')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GMM Testcase")
    plt.legend()
    plt.show()
