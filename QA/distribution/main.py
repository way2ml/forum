from numpy.core.defchararray import count
from dump_reader import *
import numpy as np 
import os 
# file_path: replace with your own
file_path = 'dump.recenter.lammpstrj'

# The line number before the first xyz coordinates,
# which equals to the line number  counting to the first line of 
# 'ITEM: ATOMS id type x y z ' in my case.
extra_line_num = 9

# Get all the dump list  
dump_list= read_dumps(file_path=file_path, extra_line_num=9)

# Dump number
print('Dump number: {}'.format(len(dump_list)))

# The first dump 
print(dump_list[0].time_step)
print(dump_list[0].xyz_df)

input()

num_bins = 250     # 把模拟盒子分成几个小盒子
num_dump = len(dump_list) # 一共有多少帧 = dump的次数
HIST = np.empty((num_dump, num_bins)) # 建立一个以行为次数，列为盒子数的一个空的数组，然后把数放进去
rowNum = 0

volume_bin = 50 * 50 * (250/num_bins)  # 单个盒子的体积
neighbour_num = 10  # 以最大粒子数的盒子为基准，隔几个盒子算做浓相范围
dilute_num = 100    # 以盒子的大小为准，确定稀相的范围
dense_box = neighbour_num*2
diluteleft_box = 25

c_de_array = np.empty(num_dump)
c_di_array = np.empty(num_dump)
counter = 0 

for dump in dump_list:  # 遍历dump list中的所有数
    # Select all particels
    XYZ = dump.xyz_df
    print('Before:', XYZ)
    XYZ = XYZ.loc[:, 2:].to_numpy() # Extract x,y,z
    print('After:', XYZ)

    X = XYZ[:0]
    # x_c = np.mean(X)  
    print('X is ', X)
    hist, bins = np.histogram(X,range=(0, 250), bins=num_bins) # 找出x的统计分布，算出每个盒子中的粒子数
    print('Distribution: {}'.format(hist))
    idx_max = np.argmax(hist)  # 找到每一帧中的最大的粒子数，并且标记它的序号
    # print('Index max: {}'.format(idx_max))
    # 调试: 用input把程序卡住，调试完成后删去
    input()
    dense_idxes = np.arange(idx_max-neighbour_num, idx_max+neighbour_num) # 确定浓相的取值范围，根据盒子的大小确定范围

    for i in range(len(dense_idxes)):     # 判断一下浓相的范围在边界的情况，超过边界的就要循环
        if dense_idxes[i] > num_bins - 1:
            dense_idxes[i] = dense_idxes[i] - num_bins
        if dense_idxes[i] < 0:
            dense_idxes[i] = dense_idxes[i] + num_bins

    # print(dense_idxes)
    concentration_dense = np.sum(hist[dense_idxes]) / (volume_bin*(dense_box)) # 计算浓相的浓度，范围内的粒子数除以范围内的体积
    # print('Concentration dense: {}'.format(concentration_dense))
    
    # input()
    diluteleft_idxes = np.arange(idx_max-dilute_num-diluteleft_box, idx_max-dilute_num)
    for i in range(len(diluteleft_idxes)):
        if diluteleft_idxes[i] > num_bins - 1:
            diluteleft_idxes[i] = diluteleft_idxes[i] - num_bins
        if diluteleft_idxes[i] < 0:
            diluteleft_idxes[i] = diluteleft_idxes[i] + num_bins
    print(diluteleft_idxes)
    
    diluteright_idxes = np.arange(idx_max+dilute_num, idx_max+dilute_num+diluteleft_box)
    for i in range(len(diluteright_idxes)):
        if diluteright_idxes[i] > num_bins - 1:
            diluteright_idxes[i] = diluteright_idxes[i] - num_bins
        if diluteright_idxes[i] < 0:
            diluteright_idxes[i] = diluteright_idxes[i] + num_bins
    print(diluteright_idxes)
    
    concentration_dilute = (np.sum(hist[diluteleft_idxes]) + np.sum(hist[diluteright_idxes]))/ (volume_bin*(diluteleft_box)*2)
    
    HIST[rowNum] = hist
    rowNum += 1
    c_de_array[counter] = concentration_dense
    c_di_array[counter] = concentration_dilute
    counter += 1
c_de_mean = np.mean(c_de_array)
c_di_mean = np.mean(c_di_array)

np.savetxt('distribution.txt', HIST, fmt='%3d')



