#!/usr/bin/env python
# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间
# 用python的tree包实现octree、KDtree

import random
import math
import numpy as np
import time
import os
import struct

from octree import OCTree
from kdtree import KDTree
from result_set import KNNResultSet, RadiusNNResultSet

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32).T

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    root_dir = '/home/wjx/Desktop/data' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)
        #build tree
        begin_t = time.time()
        octree = OCTree(point_cloud = db_np, leaf_size = leaf_size, min_extent = min_extent)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]
        
        #kNN query
        begin_t = time.time()
        knn_result_set = KNNResultSet(capacity=k)
        octree.knn_search(query, knn_result_set)
        knn_time_sum += time.time() - begin_t
        
        #RNN_query
        begin_t = time.time()
        rnn_result_set = RadiusNNResultSet(radius=radius)
        octree.rnn_fast_search(query, rnn_result_set)
        radius_time_sum += time.time() - begin_t
 
        #brute forch search
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)#linalg=linear（线性）+algebra（代数），norm则表示范数,axis=1按行向量
        nn_idx = np.argsort(diff)#sort diff
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))

    print("kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        #build tree
        begin_t = time.time()
        kd_tree = KDTree(point_cloud = db_np, leaf_size = leaf_size)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        #KNN query
        begin_t = time.time()
        knn_result_set = KNNResultSet(capacity=k)
        kd_tree.knn_search(query, knn_result_set)
        knn_time_sum += time.time() - begin_t

        #RNN query
        begin_t = time.time()
        rnn_result_set = RadiusNNResultSet(radius=radius)
        kd_tree.rnn_search(query, rnn_result_set)
        radius_time_sum += time.time() - begin_t

        #brute force search
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))



if __name__ == '__main__':
    main()