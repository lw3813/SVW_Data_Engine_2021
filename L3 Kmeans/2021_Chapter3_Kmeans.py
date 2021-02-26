# -*- coding: utf-8 -*-
# @Time: 2021/2/26 14:08
# @Author: 赵震/ BB-Driver
# @File: 2021_Chapter_3_Kmeans.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder

def main():
    data = 'car_data.csv'
    data = pd.read_csv(data, encoding='gbk')
    elbow_method(data)
    kmeans_result = kmeans_method(data, 3)
    kmeans_result.to_csv('kmeans_result.csv')
    draw_pic(kmeans_result, 3)

'''数据清洗, 均一化'''
def data_clean(data):
    train_x = data.iloc[:,:]
    min_max_scaler=preprocessing.MinMaxScaler()  #数值的特征化，默认区间为0到1
    data_min_max = min_max_scaler.fit_transform(train_x)
    return data_min_max

'''kmeans聚类'''
def kmeans_method(data, n):
#     data_title = data.iloc[:,1]
    data_train = data.iloc[:,1:]
    data_train = data_clean(data_train)
    kmeans = KMeans(n) #使用KMeans聚类
    kmeans.fit(data_train)
    predict_y = kmeans.predict(data_train)
    # 合并聚类结果，插入到原数据中
    result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
    result.rename({0:u'聚类结果'},axis=1,inplace=True)
    return result

'''聚类结果可视化'''
def draw_pic(kmeans_result, n):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    #     plt.figure(figsize=(16,16),dpi=100)

    plt.figure()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # 设置小多图之间的间隙

    plt.subplot(2, 2, 1)
    for i in range(0, n):
        x = kmeans_result[kmeans_result['聚类结果'] == i]['人均GDP']
        y = kmeans_result[kmeans_result['聚类结果'] == i]['百户拥有汽车量']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('人均GDP')
        plt.ylabel('百户拥有汽车量')

    plt.subplot(2, 2, 2)
    for i in range(0, n):
        x = kmeans_result[kmeans_result['聚类结果'] == i]['交通工具消费价格指数']
        y = kmeans_result[kmeans_result['聚类结果'] == i]['百户拥有汽车量']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('交通工具消费价格指数')
        plt.ylabel('百户拥有汽车量')

    plt.subplot(2, 2, 3)
    for i in range(0, n):
        x = kmeans_result[kmeans_result['聚类结果'] == i]['人均GDP']
        y = kmeans_result[kmeans_result['聚类结果'] == i]['城镇人口比重']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('人均GDP')
        plt.ylabel('城镇人口比重')

    plt.subplot(2, 2, 4)
    for i in range(0, n):
        x = kmeans_result[kmeans_result['聚类结果'] == i]['百户拥有汽车量']
        y = kmeans_result[kmeans_result['聚类结果'] == i]['城镇人口比重']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('百户拥有汽车量')
        plt.ylabel('城镇人口比重')

    plt.legend(loc='lower right', fontsize=6, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
    plt.savefig('Kmeans_result.png')
    plt.show()

'''手肘法判断簇数的取值'''
def elbow_method(data):
    data = data.iloc[:,1:]
    data = data_clean(data)
    sse = []
    for i in range(1, 11):
        # kmeans算法
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data,i)
        # 计算inertia簇内误差平方和
        sse.append(kmeans.inertia_)
    x = range(1, 11)
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.plot(x, sse, 'o-')
    plt.savefig('elbow_method_result.png')
    plt.show()
    return plt

if __name__ == '__main__':
    main()
