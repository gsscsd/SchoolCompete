#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: submit.py 
@time: 2017/7/28 
"""
import numpy as np
import sys
from pyspark import SparkContext

output = sys.argv[1]
inputs = sys.argv[2]

sc = SparkContext(appName='submit')
rdd = sc.textFile(inputs)

result = rdd.collect()
result_final = []

for i,item in enumerate(result):
    if item == '0.0':
        result_final.append(i + 1)

result_rdd = sc.parallelize(result_final)
result_rdd.saveAsTextFile(output)