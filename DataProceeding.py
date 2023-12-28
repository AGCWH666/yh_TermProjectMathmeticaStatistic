# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:02:55 2023

@author: 12145
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import csv
import statsmodels.api as sm
import pylab
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy import stats


def confidence_interval_u(data, sigma=-1, alpha=0.05, side_both=True):
    xb = np.mean(data)
    s = np.std(data, ddof=1)
    if  sigma > 0: # sigma已知，枢轴量服从标准正态分布
        Z = stats.norm(loc=0, scale=1.)
        if side_both: # 求双侧置信区间
            tmp = sigma/np.sqrt(len(data))*Z.ppf(1-alpha/2)
            return (xb-tmp, xb+tmp)
        else: # 单侧置信下限或单侧置信上限
            tmp= sigma/np.sqrt(len(data))*Z.ppf(1-alpha)
            return {'bottom_limit': xb-tmp, 'top_limit': xb+tmp}
    else: # sigma未知，枢轴量服从自由度为n-1的t分布
        T = stats.t(df=len(data)-1)
        if side_both:
            tmp = s/np.sqrt(len(data))* T.ppf(1-alpha/2)
            return (xb-tmp, xb+tmp)
        else:
            tmp = s/np.sqrt(len(data))* T.ppf(1-alpha)
            return {'bottom_limit': xb-tmp, 'top_limit': xb+tmp}


def confidence_interval_sigma(data, mu=-1, alpha=0.05, side_both=True):
    #xb = np.mean(data)
    #s_square = np.var(data, ddof=1)
    
    if mu > 0:
        sum_tmp = 0.0
        for i in data:
            sum_tmp = sum_tmp + (i-mu)**2
        if side_both:
            return (sum_tmp/stats.chi2.ppf(1-alpha/2, df=len(data)), sum_tmp/stats.chi2.ppf(alpha/2, df=len(data)))
        else:
            return {'bottom_limit':sum_tmp/stats.chi2.ppf(1-alpha, df=len(data)), 'top_limit':sum_tmp/stats.chi2.ppf(alpha, df=len(data))}
    else:
        tmp = (len(data)-1)*np.var(data, ddof=1)
        if side_both:
            return (tmp/stats.chi2.ppf(1-alpha/2, df=len(data)-1), tmp/stats.chi2.ppf(alpha/2, df=len(data)-1))
        else:
            return {'bottom_limit':tmp/stats.chi2.ppf(1-alpha, df=len(data)-1), 'top_limit':tmp/stats.chi2.ppf(alpha, df=len(data)-1)}



def Plots(list):
    #经验分布函数
    ecdf = ECDF(list)

    y = ecdf(list) # 计算分布函数值
    list.sort()
    y.sort()
    # 画阶梯函数之前一定要记得排序，不然就是乱七八糟的回字形
    plt.step(list, y)


    #画图
    s = pd.DataFrame(list, columns = ['rate']) 
    # 创建自定义图像
    fig = plt.figure(figsize=(10, 6))
    # 创建子图2
    ax2 = fig.add_subplot(2, 1, 2)
    # 绘制直方图
    s.hist(bins=30,alpha=0.5,ax=ax2)
    # 绘制密度图
    s.plot(kind='kde', secondary_y=True,ax=ax2)     # 使用双坐标轴
    plt.grid()      # 添加网格
     
    # 显示自定义图像
    plt.show()

#读取数据，distribute_list是记录了单日所有股票变化率的列表
filename='D:\\yh_TermProjectMathmeticaStatistic\\data\\rate_list_2099.csv'
distribute_list = []
amount_list = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:            # 将csv 文件中的数据保存到data中
        distribute_list.append(float(row[2]))           # 选择某一列加入到data数组中
        amount_list.append(float(row[3]))
log_distribute = np.log(distribute_list)


#线性回归：总量和变化率
distribute_list_abs = np.absolute(distribute_list)
amount_list = np.array(amount_list).reshape((-1, 1))
model = LinearRegression().fit(amount_list, distribute_list_abs)
pred_rate = model.predict(amount_list)
pred_rate = pd.DataFrame(pred_rate, columns=['pred_rate'])
#pred_rate.to_csv("D:\\yh_TermProjectMathmeticaStatistic\\data\\pred_rate_2099.csv", index=False)


#画图
Plots(distribute_list)
Plots(log_distribute)

#数字特征
mean = np.mean(distribute_list)
Nvar = np.var(distribute_list)


#检验分布
sm.qqplot(log_distribute, line='s')
pylab.show()

#cdf中可以指定要检验的分布，norm表示我们需要检验的是正态分布
#常见的分布包括norm,logistic,expon,gumbel等
kstest(log_distribute,cdf = "norm")

shapiro(log_distribute)


scipy.stats.normaltest(distribute_list)
scipy.stats.normaltest(log_distribute)

#参数估计
mean = np.mean(log_distribute)
ModifiyedVar = np.var(log_distribute, ddof = 1)

#区间估计
confidence_interval_u(log_distribute)
confidence_interval_sigma(log_distribute)



