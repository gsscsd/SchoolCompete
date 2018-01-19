#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: MouseSparkTrain.py
@time: 2017/7/27 
"""

import sys
from pyspark import SparkContext
import numpy as np


output = sys.argv[1]
inputs = sys.argv[2]
# inputs = '../data/dsjtzs_txfz_training.txt'
# output = '../cache/submit_test'

#统计特征值：最小值，平均值，标准差，最大值，初始值，最后值
##New Code
def optFeature(lines):
    ##这里的np
    line = [float(i) for i in lines]
    line = np.array(line)
    init = line[0]
    last = line[-1]
    std = np.std(line)
    mean = np.mean(line)
    median = np.median(line)
    max = np.max(line)
    min = np.min(line)
    return init,last,std,mean,median,max,min

##差分计算
def diff_data(line,flag):
    if len(line) == 0 :
        line.append(0)
    if len(line) == 1:
        line = line * 10
    lines = [float(i) for i in line]
    line = np.array(lines)
    diffdata = [ line[i + 1] - line[i] for i in range(len(line) - 1)]
    ###对零的处理
    if flag == 1:
        for i in range(len(diffdata)):
            if diffdata[i] == 0:
                diffdata[i] += 0.0001
    return diffdata

##计算欧式距离
def calEuclideanDistance(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

#'''获取x是否回择的评价'''
def getIsBack(xs):
    is_back = 0
    xs = np.array(xs)
    length = len(xs)
    if(length > 1):
        for index,value in enumerate(xs):
            if(index+1!=length):
                temp = xs[index+1] - xs[index]
                if(temp < 0):
                    is_back = is_back + 1
    return is_back

#获取前中后的3点数据
def getPMNSpeed(data):
    if len(data) == 0:
        data.append(0)
    if len(data) <= 3:
        data = data * 6
    lens = len(data)
    mid = lens / 2
    data = np.array(data)
    data = np.nan_to_num(data)
    return data[0:3],data[mid - 2:mid + 1],data[lens-3:lens]

#数据格式操作
def nantonum(data):
    for i in range(len(data)):
        data[i] = data[i].lstrip('[')
        data[i] = data[i].rstrip(']')
        if data[i].lower() == 'nan':
            data[i] = '0'
            print "nan"

#'''获取连续性的评价值 + 差分集合'''
def getConValue(x):
    xdiff = []
    x = np.array(x)
    convalue = 0
    length = len(x)
    if(length>1):
        for index,value in enumerate(x):
            if(index+1!=length):
                xdiff_temp = x[index+1] - x[index]
                xdiff.append(xdiff_temp)
                convalue = convalue + abs(xdiff_temp)
    else:
        convalue = 0
        xdiff=[0]
    convalue = convalue/length
    return convalue,xdiff

##真的
def fea_extra_one(lines):

    ##特征矩阵
    fea_list = []
    line = lines.split(" ")
    id = lines[0]
    label = line[-1]
    nums = line[1].strip('\n').split(';')
    aims = line[-2].strip('\n').split(',')
    xs,ys,ts = [],[],[]
    for num in nums:
        if len(num) > 0:
            xyt = num.strip('\n').split(',')
            xs.append(xyt[0])
            ys.append(xyt[1])
            ts.append(xyt[-1])
    # print(xs)
    # print(type(xs[0]))
    xs = [float(i) for i in xs]
    ys = [float(i) for i in ys]
    ts = [float(i) for i in ts]
    aims = [float(i) for i in aims]

    for i in range(len(xs)):
        xt_data = [float(xs[i]),float(ys[i])]
    aim_data = [float(aims[0]),float(aims[1])]

    ##新的特征计算
    x_distances = [calEuclideanDistance(xs[i + 1],xs[i]) for i in range(len(xs) - 1)]
    x_diff = diff_data(xs,0)
    y_diff = diff_data(ys,0)
    t_diff = diff_data(ts,1)
    xt_speed = [np.log1p(distance) - np.log1p(delta)  for (distance, delta) in zip(x_distances, t_diff)]
    xt_accu = [np.log1p(speed) - np.log1p(delta)  for (speed, delta) in zip(xt_speed, t_diff)]
    xt_angle = [np.log1p((ys[i+1] - ys[i])) - np.log1p((xs[i+1] - xs[i])) for i in range(len(xs) - 1)]
    aim_distances = [calEuclideanDistance(xt_data[i], aim_data)  for i in range(len(xt_data))]
    aim_angle = [ np.log1p(ys[0] - aims[1] ) - np.log1p( xs[0] - aims[0] )]


    ##差分特征计算
    xt_speed_diff = diff_data(xt_speed,0)
    xt_accu_diff = diff_data(xt_accu,0)
    xt_angle_diff = diff_data(xt_angle,0)
    aim_distances_diff = diff_data(aim_distances,0)
    # aim_angle_diff = diff_data(aim_angle,0)


    ##'''连续性相关特征，其他相关特征'''
    # xconvalue,xdiff = getConValue(xs)
    # yconvalue,ydiff = getConValue(ys)
    # tconvalue,tdiff = getConValue(ts)

    ##'''最后一个坐标x y距离目标x y的距离'''
    xdis_target = xs[-1] - float(aims[0])
    ydis_target = ys[-1] - float(aims[1])
    dis = np.sqrt(xdis_target**2 + ydis_target**2)


    xdis = xs[-1] - xs[0]
    ydis = ys[-1] - ys[0]
    tdis = abs(ts[-1] - ts[0])

    ###差分计算
    diffx1lines = diff_data(xs,0)
    difft1lines = diff_data(ts,1)
    ##时间的差分
    time_delta = difft1lines
    ###x,t速度
    # xtspeed = np.array(diffx1lines) / np.array(difft1lines)
    #
    # diffx2lines = diff_data(diffx1lines,0)
    # difft2lines = diff_data(difft1lines,1)
    # ##x,t加速度
    # xtaccu = np.array(diffx2lines) / np.array(difft2lines)


    count = len(xs)
    is_back = getIsBack(xs)
    # print(count)
    #x,y,t的统计特征
    xinit,xlast,xstd,xmean,xmedian,xmax,xmin = optFeature(xs)
    yinit,ylast,ystd,ymean,ymedian,ymax,ymin = optFeature(ys)
    tinit,tlast,tstd,tmean,tmedian,tmax,tmin = optFeature(ts)
    #speed,accu的统计特征
    # speedinit,speedlast,speedstd,speedmean,speedmedian,speedmax,speedmin = optFeature(xtspeed)
    # accuinit,acculast,accustd,accumean,accumedian,accumax,accumin = optFeature(xtaccu)
    time_deltainit,time_deltalast,time_deltastd,time_deltamean,time_deltamedian,time_deltamax,time_deltamin = optFeature(time_delta)
    ##other 统计特征
    xt_speedinit,xt_speedlast,xt_speedstd,xt_speedmean,xt_speedmedian,xt_speedmax,xt_speedmin = optFeature(xt_speed)
    xt_accuinit,xt_acculast,xt_accustd,xt_accumean,xt_accumedian,xt_accumax,xt_accumin = optFeature(xt_accu)
    xt_angleinit,xt_anglelast,xt_anglestd,xt_anglemean,xt_anglemedian,xt_anglemax,xt_anglemin = optFeature(xt_angle)
    aim_distancesinit,aim_distanceslast,aim_distancesstd,aim_distancesmean,aim_distancesmedian,aim_distancesmax,aim_distancesmin = optFeature(aim_distances)
    # aim_angleinit,aim_anglelast,aim_anglestd,aim_anglemean,aim_anglemedian,aim_anglemax,aim_anglemin = optFeature(aim_angle)

    xt_speed_diffinit,xt_speed_difflast,xt_speed_diffstd,xt_speed_diffmean,xt_speed_diffmedian,xt_speed_diffmax,xt_speed_diffmin = optFeature(xt_speed_diff)
    xt_accu_diffinit,xt_accu_difflast,xt_accu_diffstd,xt_accu_diffmean,xt_accu_diffmedian,xt_accu_diffmax,xt_accu_diffmin = optFeature(xt_accu_diff)
    xt_angle_diffinit,xt_angle_difflast,xt_angle_diffstd,xt_angle_diffmean,xt_angle_diffmedian,xt_angle_diffmax,xt_angle_diffmin = optFeature(xt_angle_diff)
    aim_distances_diffinit,aim_distances_difflast,aim_distances_diffstd,aim_distances_diffmean,aim_distances_diffmedian,aim_distances_diffmax,aim_distances_diffmin = optFeature(aim_distances_diff)
    # aim_angle_diffinit,aim_angle_difflast,aim_angle_diffstd,aim_angle_diffmean,aim_angle_diffmedian,aim_angle_diffmax,aim_angle_diffmin = optFeature(aim_angle_diff)

    ##前中后三点的数据
    xt_speed_first,xt_speed_mid,xt_speed_last = getPMNSpeed(xt_speed)
    xt_accu_first,xt_accu_mid,xt_accu_last = getPMNSpeed(xt_accu)
    xt_angle_first,xt_angle_mid,xt_angle_last = getPMNSpeed(xt_angle)
    xt_speed_diff_first,xt_speed_diff_mid,xt_speed_diff_last = getPMNSpeed(xt_speed_diff)
    xt_accu_diff_first,xt_accu_diff_mid,xt_accu_diff_last = getPMNSpeed(xt_accu_diff)
    xt_angle_diff_first,xt_angle_diff_mid,xt_angle_diff_last = getPMNSpeed(xt_angle_diff)

    fea_list.append(count)
    fea_list.append(is_back)
    ##x Fea
    fea_list.append(xinit)
    fea_list.append(xmean)
    fea_list.append(xmedian)
    fea_list.append(xstd)
    fea_list.append(xmax)
    fea_list.append(xmin)
    fea_list.append(xlast)
    # ##y Fea
    fea_list.append(yinit)
    fea_list.append(ymean)
    fea_list.append(ymedian)
    fea_list.append(ystd)
    fea_list.append(ymax)
    fea_list.append(ymin)
    fea_list.append(ylast)
    # ##t Fea
    fea_list.append(tinit)
    fea_list.append(tmean)
    fea_list.append(tmedian)
    fea_list.append(tstd)
    fea_list.append(tmax)
    fea_list.append(tmin)
    fea_list.append(tlast)
    ###time_delta Fea
    fea_list.append(time_deltainit)
    fea_list.append(time_deltamean)
    fea_list.append(time_deltamedian)
    fea_list.append(time_deltastd)
    fea_list.append(time_deltamax)
    fea_list.append(time_deltamin)
    fea_list.append(time_deltalast)
    ### speed accu
    # fea_list.append(speedinit)
    # fea_list.append(speedlast)
    # fea_list.append(speedmedian)
    # fea_list.append(speedmean)
    # fea_list.append(accumedian)
    # fea_list.append(accumean)
    # fea_list.append(accuinit)
    # fea_list.append(acculast)
    ###other fea
    # fea_list.append(xconvalue)
    # fea_list.append(yconvalue)
    # fea_list.append(tconvalue)
    # fea_list.append(xdis)
    # fea_list.append(ydis)
    # fea_list.append(tdis)
    fea_list.append(dis)
    fea_list.append(xdis_target)
    fea_list.append(ydis_target)

    ##new coding
    # fea_list.append(np.nan_to_num(xt_speedinit))
    # fea_list.append(np.nan_to_num(xt_speedlast))
    fea_list.append(np.nan_to_num(xt_speedmean))
    fea_list.append(np.nan_to_num(xt_speedmean))
    fea_list.append(np.nan_to_num(xt_speedmax))
    fea_list.append(np.nan_to_num(xt_speedmin))
    # fea_list.append(np.nan_to_num(xt_accuinit))
    # fea_list.append(np.nan_to_num(xt_acculast))
    fea_list.append(np.nan_to_num(xt_accumean))
    fea_list.append(np.nan_to_num(xt_accumedian))
    fea_list.append(np.nan_to_num(xt_accumax))
    fea_list.append(np.nan_to_num(xt_accumin))
    # fea_list.append(np.nan_to_num(xt_angleinit))
    # fea_list.append(np.nan_to_num(xt_anglelast))
    fea_list.append(np.nan_to_num(xt_anglemean))
    fea_list.append(np.nan_to_num(xt_anglemedian))
    fea_list.append(np.nan_to_num(xt_anglemax))
    fea_list.append(np.nan_to_num(xt_anglemin))
    fea_list.append(np.nan_to_num(aim_distancesinit))
    fea_list.append(np.nan_to_num(aim_distanceslast))
    fea_list.append(np.nan_to_num(aim_distancesmean))
    fea_list.append(np.nan_to_num(aim_distancesmedian))
    fea_list.append(np.nan_to_num(aim_distancesmax))
    fea_list.append(np.nan_to_num(aim_distancesmin))
    # fea_list.append(np.nan_to_num(aim_angleinit))
    # fea_list.append(np.nan_to_num(aim_anglelast))
    # fea_list.append(np.nan_to_num(aim_anglemean))
    # fea_list.append(np.nan_to_num(aim_anglemedian))
    # fea_list.append(np.nan_to_num(aim_anglemax))
    # fea_list.append(np.nan_to_num(aim_anglemin))
    # fea_list.append(np.nan_to_num(xt_speed_diffinit))
    # fea_list.append(np.nan_to_num(xt_speed_difflast))
    fea_list.append(np.nan_to_num(xt_speed_diffmean))
    fea_list.append(np.nan_to_num(xt_speed_diffmedian))
    fea_list.append(np.nan_to_num(xt_speed_diffmax))
    fea_list.append(np.nan_to_num(xt_speed_diffmin))
    fea_list.append(np.nan_to_num(xt_accu_diffmean))
    fea_list.append(np.nan_to_num(xt_accu_diffmedian))
    fea_list.append(np.nan_to_num(xt_accu_diffmax))
    fea_list.append(np.nan_to_num(xt_accu_diffmin))
    # fea_list.append(np.nan_to_num(xt_angle_diffinit))
    # fea_list.append(np.nan_to_num(xt_angle_difflast))
    fea_list.append(np.nan_to_num(xt_angle_diffmean))
    fea_list.append(np.nan_to_num(xt_angle_diffmedian))
    fea_list.append(np.nan_to_num(xt_angle_diffmax))
    fea_list.append(np.nan_to_num(xt_angle_diffmin))
    # fea_list.append(np.nan_to_num(aim_angle_diffinit))
    # fea_list.append(np.nan_to_num(aim_angle_difflast))
    # fea_list.append(np.nan_to_num(aim_angle_diffmean))
    # fea_list.append(np.nan_to_num(aim_angle_diffmedian))
    # fea_list.append(np.nan_to_num(aim_angle_diffmax))
    # fea_list.append(np.nan_to_num(aim_angle_diffmin))
    fea_list.append(np.nan_to_num(aim_distances_diffinit))
    fea_list.append(np.nan_to_num(aim_distances_difflast))
    fea_list.append(np.nan_to_num(aim_distances_diffmean))
    fea_list.append(np.nan_to_num(aim_distances_diffmedian))
    fea_list.append(np.nan_to_num(aim_distances_diffmax))
    fea_list.append(np.nan_to_num(aim_distances_diffmin))
    fea_list.append(np.nan_to_num(aim_angle))

    # for i in range(len(xt_speed_first)):
    #     fea_list.append(np.nan_to_num(xt_speed_first[i]))
    #     fea_list.append(np.nan_to_num(xt_speed_mid[i]))
    #     fea_list.append(np.nan_to_num(xt_speed_last[i]))
    #     fea_list.append(np.nan_to_num(xt_accu_first[i]))
    #     fea_list.append(np.nan_to_num(xt_accu_mid[i]))
    #     fea_list.append(np.nan_to_num(xt_accu_last[i]))
    #     fea_list.append(np.nan_to_num(xt_angle_first[i]))
    #     fea_list.append(np.nan_to_num(xt_angle_mid[i]))
    #     fea_list.append(np.nan_to_num(xt_angle_last[i]))
        # fea_list.append(np.nan_to_num(xt_speed_diff_first[i]))
        # fea_list.append(np.nan_to_num(xt_speed_diff_mid[i]))
        # fea_list.append(np.nan_to_num(xt_speed_diff_last[i]))
        # fea_list.append(np.nan_to_num(xt_accu_diff_first[i]))
        # fea_list.append(np.nan_to_num(xt_accu_diff_mid[i]))
        # fea_list.append(np.nan_to_num(xt_accu_diff_last[i]))
        # fea_list.append(np.nan_to_num(xt_angle_diff_first[i]))
        # fea_list.append(np.nan_to_num(xt_angle_diff_mid[i]))
        # fea_list.append(np.nan_to_num(xt_angle_diff_last[i]))
    #添加标签列
    fea_list.append(label)
    #终极特征矩阵

    # print(len(fea_list))

    fea_str_list = [str(item) for item in fea_list]
    nantonum(fea_str_list)
    print(len(fea_str_list))
    fea_str = ' '.join(fea_str_list)

    return fea_str

#
########
########程序开始的地方
#####
sc = SparkContext(appName="train")
rdd = sc.textFile(inputs)
result = rdd.map(fea_extra_one)
# print(result.collect())
print(result.count())
print(rdd.take(10))
result.saveAsTextFile(output)