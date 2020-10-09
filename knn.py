# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:07:55 2020

@author: Louis Zhang
"""
'''
此程序只使用pandas实现KNN算法
'''
#导入pandas
import pandas as pd 
#加载数据
def load_data(filename):
    '''生成训练集和测试集'''
    data = pd.read_csv(filename, engine = 'python', encoding = 'UTF-8')
    return data

#生成训练集和测试集
def data_proc(data):
    '''随机抽样pd.DataFrame.sample'''
    train_data = data.sample(frac = 0.7)
    #isin函数判断train_data.index是否在data.index中,并返回布尔型
    test_data = data[~data.index.isin(train_data.index)]
    #修改行索引,不需要修改测试集的行索引,因为测试集没有用到
    train_data.index = range(len(train_data))
    return train_data, test_data

#求距离
def dist(x1, x2):
    '''求两个向量的欧氏距离'''
    s = 0
    for i in data.columns[2:]:
        s += (float(x1[i]) - float(x2[i])) ** 2
    return s ** 0.5

#获取样本点x的类别
def get_classification(train_data, x, k):
    '''
    k为指定最近的k个点
    设定空列表d用于存放距离
    当数据框使用默认索引时，使用iloc属性访问一行
    append函数用于向列表追加数据
    '''
    d = []
    for i in range(len(train_data)):
        distance = dist(train_data.iloc[i], x)
        d.append(distance)
    #将列表转换为数据框
    d = pd.DataFrame(d)
    #定义列名
    d.columns = ['distance']
    #给距离排序，然后获取前k个距离
    d_topk = d.sort_values(by = ['distance'], ascending = True)[:k]
    #前k个距离的索引对应训练集中的索引
    ##提取这k个数据对应的类
    topk_classification = train_data[train_data.index.isin(d_topk.index)]['diagnosis_result']
    #计数
    count1 = 0
    count2 = 0
    for i in topk_classification:
        if i == 'B':
            count1 += 1
        else :
            count2 += 1
    #多数表决
    if count1 >= count2:
        x_diagnosis_result = 'B'
    else:
        x_diagnosis_result = 'M'
    return x_diagnosis_result

#测试模型准确率
def model_test(train_data, test_data):
    correct_num = 0 
    for i in range(len(test_data)):
        diagnosis_result = get_classification(train_data, test_data.iloc[i], 3)
        if diagnosis_result == test_data['diagnosis_result'].iloc[i]:
            correct_num += 1
    accur = correct_num / len(test_data)
    return accur

#执行程序
filename = 'C:\\Users\\Louis Zhang\\Desktop\\PY EXC\\统计学习方法\\KNN\\癌症\\Prostate_Cancer.csv'
data = load_data(filename)
train_data, test_data = data_proc(data)
accur = model_test(train_data, test_data)
print('模型的准确率为：')
print('accur =  %d' % (accur * 100 ), '%')
    

