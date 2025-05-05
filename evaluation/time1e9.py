# time1e9.py
# 用法：获取MH04_GT.txt，丢掉第一行，将第一列除以1e9，生成MH04_GT_1E9.txt，存在的问题有这样子除出来的浮点型丢了一两个单位的精度，不过实际测试似乎问题不大

import sys
pathin = "./Ground_truth/EuRoC_left_cam/MH03_GT.txt"   #数据来源
pathout = "./Ground_truth/EuRoC_left_cam/MH03_GT_1E9.txt"
import os

f = open(pathin,'r',encoding = 'UTF-8')
open(pathout,"w+").close()

list1 = [5000]

line = f.readline() # MH01_GT.txt第一行为参数注释，丢掉就行
# line = f.readline()

# -----单次调试----
'''
a = line.split(",")
b = a[0] #第一列数
c = a[1:] # c is a list

list1.append(b)
list1.append(c)

if b is not None:
    b = float(b) / 1e9
    with open(pathout,"a") as f_out:
        # for i in list1:
        first = str(b)
        f_out.write(first + ' ')
        for j in range(len(c)-1):
            f_out.write(str(c[j]) + ' ')
        f_out.write(str(c[(len(c)-1)]))
    f_out.close()
'''
# ----

while line is not None and line !='':

    line = f.readline()
    a = line.split(",")
    b = a[0] #第一列数
    c = a[1:] # c is a list

    list1.append(b)
    list1.append(c)

    # with open(pathout,"w") as f_out:
    #     f_out.write(str(list1))
    # f_out.close()
    if b != '':
        b = float(b) / 1e9
        with open(pathout,"a") as f_out:
            first = str(b)
            f_out.write(first + ',')
            for j in range(len(c)-1):
                f_out.write(str(c[j]) + ',')
            f_out.write(str(c[(len(c)-1)]))
        f_out.close()
    # print(b)


f.close()

print("write done!")