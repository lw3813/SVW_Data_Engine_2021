# -*- coding: utf-8 -*- 
# @Time: 2021/2/5 8:58
# @Author: 赵震/ BB-Driver
# @File: 2021_Chapter_2_Titanic.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, plot_importance

#主程序
def main():
    #导入数据
    dir_train = 'train.csv'
    dir_test = 'test.csv'
    train_data = data_input(dir_train)
    test_data = data_input(dir_test)
    #数据清洗
    train_data = data_clean(train_data)
    test_data = data_clean(test_data)
    #选取特征, 及非数值特征的数值化
    train_features = label_encode(train_data)
    test_features = label_encode(test_data)
    train_labels = train_data['Survived']
    #数据均一化
    train_features = MinMax(train_features)
    test_features = MinMax(test_features)
    #切分数据集用于神经网络计算
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1)
    #训练神经网络NN, 并返回accuracy
    NN_results = NN(x_train, x_test, y_train, y_test,epochs=200)
    #各种机器模型算法计算
    Classifiers, scores = multi_machine_learning(train_features,train_labels)
    #整理NN与ML计算结果
    mode = pd.DataFrame(scores, index=Classifiers, columns=['score']).sort_values(by='score', ascending=False)
    mode.loc['Neural Network'] = NN_results[1]   #添加NN结果
    final_result = mode.sort_values('score', ascending=False)
    print('-'*50+'Results'+'-'*50)
    print(final_result)


#导入数据
def data_input(dir):
    df = pd.read_csv(dir)
    return df

#清理数据
def data_clean(df):
    # 使用平均年龄来填充年龄中的nan值
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    # 使用票价的均值填充票价中的nan值
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    # 使用登录最多的港口来填充登录港口的nan值
    df['Embarked'].fillna('S', inplace=True)
    return df

#文字内容数据化
def label_encode(data,features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']):
    le = LabelEncoder()
    features = data[features]
    features['Sex'] = le.fit_transform(features['Sex'])
    features['Embarked'] = le.fit_transform(features['Embarked'])
    return features


#数据均一化
def MinMax(features):
    mm = MinMaxScaler()
    features = mm.fit_transform(features)
    return features


#调用各类机器学习算法
def multi_machine_learning(train_features, train_labels):
    clf_rf = RandomForestClassifier()
    clf_et = ExtraTreesClassifier()
    clf_bc = BaggingClassifier()
    clf_ada = AdaBoostClassifier()
    clf_dt = DecisionTreeClassifier()
    clf_xg = XGBClassifier()
    clf_lr = LogisticRegression()
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_svm = SVC()
    Classifiers = ['RandomForest', 'ExtraTrees', 'Bagging', 'AdaBoost', 'DecisionTree', 'XGBoost', 'LogisticRegression',
                   'KNN', 'SVM']
    scores = []
    models = [clf_rf, clf_et, clf_bc, clf_ada, clf_dt, clf_xg, clf_lr, clf_knn, clf_svm]
    for model in models:
        score = cross_val_score(model, train_features, train_labels, scoring='accuracy', cv=10, n_jobs=-1).mean()
        scores.append(score)
    #整理罗列结果
    dic_raw = zip(Classifiers, scores)
    ml_results = dict((name, value) for name, value in dic_raw)
    ml_results
    print(ml_results)
    # print('ML算法:{name},分数:{score_number}'.format(name=Classifiers,score_number=scores))
    return Classifiers,scores

#调用神经网络
def NN(x_train,x_test,y_train,y_test,epochs= 40):
    # 搭建模型
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=[7]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        # keras.layers.Dense(64, activation='relu'),
        # keras.layers.Dense(128,activation='relu'),
        # keras.layers.Dense(64,activation='relu'),
        # keras.layers.Dense(32,activation='relu'),
        # keras.layers.Dense(20,activation='relu'),
        keras.layers.Dense(2, activation='softmax')])
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')
    # 模型训练
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test))
    results = model.evaluate(x_test, y_test)
    print(results)
    return results

if __name__ == '__main__':
    main()