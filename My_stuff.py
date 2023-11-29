# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:41:08 2023

@author: 12145
"""

import baostock as bs
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np


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



#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

data_list = []
rate_list = []


#所有天的所有股票
pass


#单天的所有股票
#修改右边界700000
for i in range(600000, 700000):
    code = "sh." + str(i)
    rs = bs.query_history_k_data_plus(code,
                                      "date,code,open,close,amount",
                                      start_date='2022-7-01', end_date='2022-8-02',
                                      frequency="m", adjustflag="3")
    print(i)
    
    #单天单股    
    while (rs.error_code == '0') & rs.next():
        content = rs.get_row_data()
        data_list.append(content)
        #排除空串
        if len(content[2]) == 0 or len(content[3]) == 0 or len(content[4]) == 0: 
            break
        rate = content[:2]
        
        rate.append(float(content[3])/float(content[2]))
        rate.append(content[4])
            
        rate_list.append(rate)

#获取验证分布的数据
distribute_list = []

for i in rate_list:
    distribute_list.append(i[2])

log_distribute = np.log(distribute_list)

#画图
Plots(log_distribute)

#检验分布
scipy.stats.normaltest(distribute_list)
scipy.stats.normaltest(log_distribute)


#数字特征
mean = np.mean(distribute_list)
var = np.var(distribute_list, ddof = 1)




#看结果
result = pd.DataFrame(data_list, columns=rs.fields)
rate_result = pd.DataFrame(rate_list, columns=['date', 'code', 'rate', 'amount'])



#### 结果集输出到csv文件 ####   
result.to_csv("D:\\yh_TermProjectMathmeticaStatistic\\data_list_2099.csv", index=False)
rate_result.to_csv("D:\\yh_TermProjectMathmeticaStatistic\\rate_list_2099.csv", index=False)


print(result)
print(rate_result)

#### 登出系统 ####
bs.logout()